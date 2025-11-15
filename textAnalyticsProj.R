# 0. Libraries
library(readr)
library(dplyr)
library(stringr)
library(udpipe)
library(tokenizers)
library(text2vec)
library(igraph)
library(ggraph)
library(ggplot2)
library(visNetwork)
library(tidyr)
library(textclean)
library(widyr)

set.seed(42)

# 1. Load dataset (change filename if needed)
library(jsonlite)
con <- file("/Users/keziah/Downloads/arxiv-metadata-oai-snapshot.json", "r")

# Read first 5000 lines only
# Read and parse JSON safely
lines <- readLines(con, n = 5000)
close(con)

subset_data <- lapply(lines, function(line) {
  tryCatch({
    obj <- fromJSON(line)
    tibble(
      id = obj$id %||% NA,
      title = obj$title %||% NA,
      authors = if (is.list(obj$authors)) paste(unlist(obj$authors), collapse = ", ") else as.character(obj$authors),
      categories = obj$categories %||% NA,
      abstract = obj$abstract %||% NA
    )
  }, error = function(e) NULL)
})

df <- bind_rows(subset_data)


# Quick fallback if file not found: create tiny toy dataset (uncomment to test)
if(nrow(df) == 0) {
  df <- tibble(
    id = c("p1","p2","p3"),
    title = c("Deep Learning for Images", "Graph Neural Networks", "Vision & Transformers"),
    authors = c("Alice A., Bob B.", "Charlie C., Dana D.", "Alice A., Ella E."),
    categories = c("cs.CV", "cs.LG", "cs.CV cs.LG"),
    abstract = c(
      "We propose a deep convolutional network for image classification. The network achieves high accuracy.",
      "We introduce a graph neural network (GNN) method for node classification and show improved performance.",
      "Transformers achieve state-of-the-art results on vision tasks by using patch embeddings and attention."
    )
  )
}

# Optionally limit size for development
max_papers <- 500    # change to larger number as your machine allows
df <- df %>% slice(1:min(nrow(df), max_papers))

# 2. Preprocessing helper functions
clean_text <- function(x) {
  x %>%
    replace_contraction() %>%
    replace_symbol() %>%
    replace_url() %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

df <- df %>% mutate(abstract = sapply(abstract, clean_text))

# 3. UDPIPE model download and annotation (dependency parse + POS + lemmas)
ud_model_file <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(ud_model_file$file_model)

# annotate abstracts sentence-wise
annotations <- udpipe_annotate(ud_model, x = df$abstract, doc_id = df$id)
annotations <- as.data.frame(annotations)

# 4. Extract subject-verb-object triples (simple heuristic)
# For each sentence, find verbs and their nominal subjects / direct objects
library(purrr)

extract_svo <- function(sent_df) {
  svos <- list()
  verbs <- sent_df %>% filter(upos %in% c("VERB", "AUX"))
  if (nrow(verbs) == 0) return(NULL)
  
  for (i in seq_len(nrow(verbs))) {
    v <- verbs[i, ]
    v_id <- v$token_id  # ✅ fixed column name
    subs <- sent_df %>%
      filter(head_token_id == v_id & dep_rel %in% c("nsubj", "nsubjpass")) %>%
      pull(token)
    objs <- sent_df %>%
      filter(head_token_id == v_id & dep_rel %in% c("obj", "dobj", "iobj", "obl")) %>%
      pull(token)
    
    if (length(subs) > 0 && length(objs) > 0) {
      for (s in subs) for (o in objs) {
        svos[[length(svos) + 1]] <- tibble(
          subject = s,
          relation = v$lemma,
          object = o
        )
      }
    }
  }
  if (length(svos) == 0) return(NULL)
  bind_rows(svos)
}

# Split annotations into sentences
sents <- annotations %>%
  group_by(doc_id, paragraph_id, sentence_id) %>%
  group_split()

# Apply safely so one bad sentence doesn’t break everything
svo_list <- map(sents, ~ tryCatch(extract_svo(.x), error = function(e) NULL))
svo_list <- compact(svo_list)  # remove NULLs

# Combine and attach doc IDs
svo_df <- map2_dfr(svo_list, sents[seq_along(svo_list)], function(svo, sent) {
  if (is.null(svo)) return(NULL)
  svo$doc_id <- unique(sent$doc_id)
  svo
})

# Quick preview
print(head(svo_df))

# 5. Extract noun phrases as entities (simple pattern: consecutive NOUN/PROPN tokens)
extract_entities_from_sentence <- function(sent_df) {
  ents <- list()
  toks <- sent_df
  i <- 1
  while(i <= nrow(toks)) {
    if(toks$upos[i] %in% c("NOUN","PROPN","ADJ","NUM")) {
      # accumulate a phrase while POS fits (NOUN/PROPN/ADJ/NUM)
      j <- i
      while(j <= nrow(toks) && toks$upos[j] %in% c("NOUN","PROPN","ADJ","NUM")) j <- j+1
      phrase <- toks$token[i:(j-1)] %>% paste(collapse = " ")
      ents[[length(ents)+1]] <- tibble(entity = phrase, doc_id = unique(toks$doc_id))
      i <- j
    } else i <- i+1
  }
  if(length(ents)==0) return(NULL)
  bind_rows(ents)
}

entities_list <- lapply(sents, extract_entities_from_sentence)
entities <- bind_rows(entities_list) %>% distinct()

# 6. Extract keywords using text2vec's tf-idf (to later create edges "paper uses keyword")
it <- itoken(df$abstract, progressbar = FALSE)
v <- create_vocabulary(it, stopwords = stopwords::stopwords("en"))
v <- prune_vocabulary(v, term_count_min = 3) # adjust thresholds
vectorizer <- vocab_vectorizer(v)
dtm <- create_dtm(it, vectorizer)
tfidf <- TfIdf$new()
tfidf_dtm <- fit_transform(dtm, tfidf)
# compute top keywords per document
top_n_keywords <- function(row_vec, vocab, n = 5) {
  idx <- order(row_vec, decreasing = TRUE)[1:n]
  words <- vocab$term[idx]
  words[!is.na(words)]
}
vocab_df <- v %>% arrange(desc(term_count)) # helpful mapping
# Get top 5 keywords per doc
keywords_per_doc <- apply(tfidf_dtm, 1, function(r) {
  ord <- order(r, decreasing = TRUE)
  top <- ord[1:5]
  names(r)[top]
})
# make dataframe
kw_df <- tibble(doc_id = df$id, keywords = I(lapply(1:nrow(df), function(i) {
  # handle if vocabulary shorter
  vec <- tfidf_dtm[i,]
  ord <- order(vec, decreasing = TRUE)
  top <- ord[1:5]
  colnames(tfidf_dtm)[top]
})))

# 7. Build triples dataframe (symbolic triples)
triples <- tibble(subject = character(), relation = character(), object = character())

# Add author -> wrote -> paper
for(i in seq_len(nrow(df))) {
  paper <- df$title[i]
  auths <- str_split(df$authors[i], ",")[[1]] %>% str_trim() %>% unique()
  for(a in auths) {
    triples <- triples %>% add_row(subject = a, relation = "wrote", object = paper)
  }
  # categories
  cats <- str_split(df$categories[i], " ")[[1]] %>% str_trim()
  for(c in cats) if(nchar(c)>0) triples <- triples %>% add_row(subject = c, relation = "category_of", object = paper)
  # keywords
  kws <- kw_df$keywords[[i]]
  for(k in kws) if(!is.na(k) && nchar(k)>0) triples <- triples %>% add_row(subject = paper, relation = "uses_keyword", object = k)
}

# Add SVO triples extracted from sentences: subject-rel-object for a given doc -> attach doc title
if(nrow(svo_df) > 0) {
  for(i in seq_len(nrow(svo_df))) {
    docid <- svo_df$doc_id[i]
    paper_title <- df$title[df$id == docid]
    if(length(paper_title) == 0) next
    triples <- triples %>% add_row(subject = svo_df$subject[i], relation = paste0("verb_", svo_df$relation[i]), object = svo_df$object[i])
    # link the fact to the paper
    triples <- triples %>% add_row(subject = paper_title, relation = "mentions", object = svo_df$subject[i])
    triples <- triples %>% add_row(subject = paper_title, relation = "mentions", object = svo_df$object[i])
  }
}

# Add entity mentions as paper -> mentions -> entity
if(nrow(entities) > 0) {
  for(i in seq_len(nrow(entities))) {
    paper_title <- df$title[df$id == entities$doc_id[i]]
    if(length(paper_title)==0) next
    triples <- triples %>% add_row(subject = paper_title, relation = "mentions", object = entities$entity[i])
  }
}

# 8. Neural-style semantic similarity (embeddings) using text2vec: doc vectors by averaging GloVe-like word vectors
# Create GloVe embeddings from corpus (small demo) OR use fasttext pre-trained (not included); so we build corpus-level glove
tokens <- word_tokenizer(tolower(df$abstract))
it2 <- itoken(tokens, progressbar = FALSE)
vocab2 <- create_vocabulary(it2)
vocab2 <- prune_vocabulary(vocab2, term_count_min = 2)
vectorizer2 <- vocab_vectorizer(vocab2)
tcm <- create_tcm(it2, vectorizer2, skip_grams_window = 5L)
glove <- GloVe$new(rank = 50, x_max = 10)  # 50-dim small
wvecs <- glove$fit_transform(tcm, n_iter = 20)
# get word vectors matrix
word_vectors <- wvecs + t(glove$components)
# compute doc vectors by averaging word vectors
doc_vectors <- sapply(tokens, function(toklist) {
  vecs <- word_vectors[toklist[toklist %in% rownames(word_vectors)], , drop = FALSE]
  if(is.null(dim(vecs))) {
    if(length(vecs) == 0) return(rep(0, ncol(word_vectors)))
    return(as.numeric(vecs))
  } else {
    return(colMeans(vecs))
  }
})
doc_vectors <- t(doc_vectors) # rows = docs

# Compute cosine similarities and add "related_to" edges for pairs with similarity > threshold
# Compute cosine similarities
cos_sim <- sim2(x = doc_vectors, y = doc_vectors, method = "cosine", norm = "l2")
diag(cos_sim) <- 0

# Filter out invalid or missing values
cos_sim[is.na(cos_sim)] <- 0

# Threshold for "related_to" edges
sim_threshold <- 0.65

# Get paper titles
paper_titles <- df$title

# Find all pairs above threshold
pairs <- which(cos_sim >= sim_threshold, arr.ind = TRUE)

# Avoid duplicates (i->j and j->i)
pairs <- pairs[pairs[,1] < pairs[,2], , drop = FALSE]

# Add as triples
for (k in seq_len(nrow(pairs))) {
  i <- pairs[k, 1]
  j <- pairs[k, 2]
  triples <- triples %>%
    add_row(
      subject = paper_titles[i],
      relation = "related_to",
      object = paper_titles[j]
    )
}


# 9. Symbolic inference rule (simple): if two papers share >= 2 keywords -> same_research_area
# Build keywords per paper from triples
paper_kw <- triples %>% filter(str_detect(relation, "uses_keyword")) %>% rename(paper = subject, keyword = object) %>% group_by(paper) %>%
  summarise(keywords = list(unique(keyword)))
# naive pairwise comparison
papers <- paper_kw$paper
for(i in seq_along(papers)) {
  for(j in seq_along(papers)) {
    if(i >= j) next
    k1 <- paper_kw$keywords[[i]]
    k2 <- paper_kw$keywords[[j]]
    if(length(intersect(k1,k2)) >= 2) {
      triples <- triples %>% add_row(subject = papers[i], relation = "same_research_area", object = papers[j])
    }
  }
}

# 10. Build igraph knowledge graph
# Create unique nodes and edges
edges <- triples %>% rename(from = subject, to = object, label = relation) %>% select(from, to, label) %>% distinct()
nodes <- tibble(id = unique(c(edges$from, edges$to))) %>% mutate(label = id)

g <- graph_from_data_frame(edges, directed = TRUE, vertices = nodes)

# add node types heuristically: author if commas? paper if matches df$title
V(g)$type <- ifelse(V(g)$name %in% df$title, "paper",
                    ifelse(V(g)$name %in% unlist(str_split(paste(df$authors, collapse = ","), ",")), "author",
                           ifelse(V(g)$name %in% unlist(kw_df$keywords), "keyword", "entity")))

# 11. Simple graph queries examples
# a) authors who wrote > 1 paper in this subset
V(g)$name <- trimws(V(g)$name)

# Get author nodes
author_nodes <- V(g)[V(g)$type == "author"]

# Filter valid ones
valid_authors <- author_nodes$name[author_nodes$name %in% V(g)$name]

# Count number of papers per author safely
authors_multi <- sapply(valid_authors, function(a) {
  nb <- tryCatch(neighbors(g, a, mode = "out"), error = function(e) NULL)
  if (is.null(nb)) return(0)
  sum(V(g)[nb]$type == "paper")
})

authors_multi_df <- tibble(
  author = valid_authors,
  papers_written = authors_multi
) %>%
  arrange(desc(papers_written))

print(head(authors_multi_df, 10))

# b) find papers related to a given paper
paper_example <- V(g)$name[V(g)$type == "paper"][1]
related_papers <- neighbors(g, paper_example, mode = "all")$name
cat("Related to paper:", paper_example, "\n")
print(related_papers)

# 12. Visualize (ggraph)
# subgraph: focus on papers + authors only to keep plot readable
sub_v <- V(g)[V(g)$type %in% c("paper","author")]
sub_g <- induced_subgraph(g, vids = sub_v)
ggraph(sub_g, layout = "fr") +
  geom_edge_link(aes(label = label), alpha = 0.4, show.legend = FALSE) +
  geom_node_point(aes(filter = (type=="paper"), size = 3), color = "skyblue") +
  geom_node_point(aes(filter = (type=="author"), size = 2), color = "orange") +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  theme_void()

# 13. Interactive visualization (visNetwork) - pick top N nodes
vis_nodes <- data.frame(id = V(g)$name, label = V(g)$name, group = V(g)$type, title = V(g)$name, stringsAsFactors = FALSE)
vis_edges <- data.frame(from = as.character(edges$from), to = as.character(edges$to), label = edges$label, stringsAsFactors = FALSE)
visNetwork(vis_nodes, vis_edges) %>%
  visLegend() %>%
  visOptions(highlightNearest = TRUE, selectedBy = "group")

# 14. Save triples for inspection
write_csv(triples, "kg_triples.csv")
cat("Triples written to kg_triples.csv\n")

# 1️⃣ Graph-level metrics
num_nodes <- vcount(g)
num_edges <- ecount(g)
density <- edge_density(g)
avg_degree <- mean(degree(g))
num_components <- components(g)$no
diameter_val <- diameter(g, directed = TRUE, weights = NA)

cat("GRAPH STRUCTURAL METRICS:\n")
cat("• Number of nodes:", num_nodes, "\n")
cat("• Number of edges:", num_edges, "\n")
cat("• Graph density:", round(density, 3), "\n")
cat("• Average degree:", round(avg_degree, 2), "\n")
cat("• Number of connected components:", num_components, "\n")
cat("• Graph diameter:", diameter_val, "\n\n")

# 2️⃣ Node-level centrality measures
degree_cent <- degree(g, mode = "all", normalized = TRUE)
bet_cent <- betweenness(g, directed = TRUE, normalized = TRUE)
clos_cent <- closeness(g, mode = "all", normalized = TRUE)

centrality_df <- tibble(
  node = V(g)$name,
  type = V(g)$type,
  degree_centrality = degree_cent,
  betweenness = bet_cent,
  closeness = clos_cent
) %>%
  arrange(desc(degree_centrality))

cat("TOP 5 CENTRAL NODES:\n")
print(head(centrality_df, 5))
cat("\n")

# 3️⃣ Edge-type distribution
edge_types <- table(E(g)$relation)
cat("EDGE RELATION DISTRIBUTION:\n")
print(edge_types)
cat("\n")

# Semantic validation (for 'related_to' edges)

if (exists("cos_sim") && exists("paper_titles")) {
  
  # Convert igraph edges to dataframe
  edge_df <- as_data_frame(g, what = "edges")
  
  # Detect correct edge attribute column name (relation / label)
  attr_col <- if ("relation" %in% colnames(edge_df)) "relation" else 
    if ("label" %in% colnames(edge_df)) "label" else NA
  
  if (!is.na(attr_col)) {
    # Filter only 'related_to' edges
    related_edges <- edge_df %>%
      filter(.data[[attr_col]] == "related_to")
    
    # Compute cosine similarities for related_to pairs
    related_scores <- mapply(function(a, b) {
      i <- match(a, paper_titles)
      j <- match(b, paper_titles)
      if (!is.na(i) && !is.na(j)) cos_sim[i, j] else NA
    }, related_edges$from, related_edges$to)
    
    # Print stats
    cat("SEMANTIC SIMILARITY STATS (for 'related_to' edges):\n")
    cat("Mean cosine similarity:", round(mean(related_scores, na.rm = TRUE), 3), "\n")
    cat("SD cosine similarity:", round(sd(related_scores, na.rm = TRUE), 3), "\n")
    cat("Number of 'related_to' edges evaluated:", sum(!is.na(related_scores)), "\n")
    
  } else {
    cat(" No edge attribute found for relation/label — cannot validate semantic edges.\n")
  }
  
} else {
  cat(" Missing 'cos_sim' matrix or 'paper_titles' — skipping semantic validation.\n")
}

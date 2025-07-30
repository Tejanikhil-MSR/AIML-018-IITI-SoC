# from PDFReader import PDFSummarizer
# import pytesseract

# ROOT_DIR = r"/home/saranshvashistha/workspace/AIML-018-IITI-SoC/data/RAW_PDF_DOCS"
# SAVE_DIR = r"../data/Summarized_PDFs"
# MODEL_ENDPOINT = "http://0.0.0.0:11434/api/generate"
# model = "llama3"
# summarizer = PDFSummarizer(model_endpoint=MODEL_ENDPOINT, model=model, dataset_root_dir=ROOT_DIR)

# summarizer.summarize_all_pdfs_and_save(SAVE_DIR)

################## Document segregation ##################

# from DocumentClustering import cluster_documents_with_hdbscan, generate_cluster_names, organize_documents_by_cluster

# clustered_documents = cluster_documents_with_hdbscan(folder_path="../data/Summarized_PDFs", min_cluster_size=5, min_samples=5,
                                                    #  reduce_dim=True, n_components=100, visualize=False)

# cluster_labels = generate_cluster_names(clustered_documents, "../data/Summarized_PDFs", top_n=5)

# organize_documents_by_cluster(folder_path="../data/Summarized_PDFs", cluster_dict=clustered_documents, cluster_names=cluster_labels)

################# Keyword extractor ######################

from KeywordExtractor import extract_class_keywords

extract_class_keywords(root_folder="../Data/Summarized_PDFs/", n_keywords=20, 
                       min_df=2,             # Term must appear in >=2 classes
                       max_df=0.6            # Term must appear in <=60% of classes
                      )
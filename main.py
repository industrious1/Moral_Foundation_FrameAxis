if __name__ == '__main__':
    print("Running FrameAxis Moral Foundations scores")
    args = parse_arguments()

    print("Arguments parsed:", args)
    
    IN_PATH = args.input_file
    DICT_TYPE = args.dict_type
    print(f"Input file path: {IN_PATH}, Dictionary type: {DICT_TYPE}")

    if DICT_TYPE not in ["emfd", "mfd", "mfd2", "customized"]:
        raise ValueError(
            f'Invalid dictionary type received: {DICT_TYPE}, dict_type must be one of \"emfd\", \"mfd\", \"mfd2\", \"customized\"')
    
    OUT_CSV_PATH = args.output_file
    DOCS_COL = args.docs_colname
    print(f"Output file path: {OUT_CSV_PATH}, Document column name: {DOCS_COL}")

    if args.word_embedding_model is not None:
        W2V_PATH = args.word_embedding_model
        print(f"Loading word embedding model from {W2V_PATH}")
        model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
    else:
        print('Downloading word embedding model: word2vec-google-news-300')
        import gensim.downloader
        model = gensim.downloader.load('word2vec-google-news-300')

    print("Word embedding model loaded successfully.")

    try:
        data = pd.read_csv(IN_PATH, lineterminator='\n')
        print(f"Data loaded successfully. Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

    try:
        fa = FrameAxis(mfd=DICT_TYPE, w2v_model=model)
        print("FrameAxis instance created successfully.")
        
        mf_scores = fa.get_fa_scores(df=data, doc_colname=DOCS_COL, tfidf=False, format="virtue_vice",
                                     save_path=OUT_CSV_PATH)
        print("Moral Foundation scores calculated and saved successfully.")
    except Exception as e:
        print(f"Error during FrameAxis processing: {e}")
        raise e

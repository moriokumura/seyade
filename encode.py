from shared import *

def main(args):
    
    encode_path = args.encode_path
    model_name = args.model_name
    
    print("Loading Encode Data from {}...".format(model_path(model_name, encode_path)))
    labels_encode, docs_encode = load_docs(model_path(model_name, encode_path))
    print("Loaded {} samples.".format(len(docs_encode)))
    
    print("Loading dictionaries...")
    with io.open(model_path(model_name, CHARS_DICT_FILE), 'rb') as f:
        chars_dict = pkl.load(f)
    with io.open(model_path(model_name, LABELS_DICT_FILE), 'rb') as f:
        labels_dict = pkl.load(f)
    
    n_chars = len(chars_dict) + 1
    n_classes = len(labels_dict) + 1
    print("Model contains {} different characters.".format(n_chars - 1))
    print("Model contains {} different classes.".format(n_classes - 1))
    
    print("Vectorizing docs and labels...")
    x_chars, x_masks = vectorize_x(docs_encode, chars_dict)
    y = vectorize_y(labels_encode, labels_dict)
    
    print("Loading model from {}...".format(model_path(model_name, BEST_MODEL_FILE)))
    model = Seyade(model_name)
    model.build(n_chars, n_classes)
    model.load_params()
    
    print("Encoding and predicting...")
    
    result = {}
    result['docs'] = docs_encode
    result['targets'] = y
    result['prediction_scores'] = []
    result['embeddings'] = []
    binary_prdedictions = []
    binary_targets = []
    
    batch_count = 0
    time_encode_start = time.time()
    
    # batch encoding
    for x_chars_batch, x_masks_batch, y_batch in batches(x_chars, x_masks, y):
        
        continuous = model.predict(x_chars_batch, x_masks_batch)
        result['prediction_scores'].extend(continuous)
        result['embeddings'].extend(model.embed(x_chars, x_masks))
        binary_prdedictions.extend(np.around(continuous).astype(int).tolist())
        binary_targets.extend(y_batch.tolist())
        
        batch_count += 1
        if batch_count % DISPLAY_FREQUENCY == 0:
            print("Batch {} Time {}".format(batch_count, timedelta(seconds=(time.time() - time_encode_start))))
    
    print("Saveing result...")
    model.save_result(result, ENCODE_RESULT_FILE)
    
    accuracy = jaccard_similarity_score(np.asarray(binary_targets), np.asarray(binary_prdedictions))
    precision = precision_score(np.asarray(binary_targets), np.asarray(binary_prdedictions), average='micro')
    recall = recall_score(np.asarray(binary_targets), np.asarray(binary_prdedictions), average='micro')
    print("Result: accuracy {:.5f} precision {:.5f} recall {:.5f}".format(accuracy, precision, recall))
    print("Total Time = {}".format(timedelta(seconds=time.time() - time_encode_start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode documents')
    parser.add_argument('--model_name', default='example', help='model dir name')
    parser.add_argument('--encode_path', default='encode.txt', help='encode file name')
    args = parser.parse_args()
    main(args)

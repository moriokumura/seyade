from shared import *

def main(args):
    
    train_path = args.train_path
    test_path = args.test_path
    model_name = args.model_name
    
    print("Loading Training Data from {}...".format(model_path(model_name, train_path)))
    labels_train, docs_train = load_docs(model_path(model_name, train_path))
    print("Loaded {} training samples.".format(len(docs_train)))

    print("Loading Test Data from {}...".format(model_path(model_name, test_path)))
    labels_test, docs_test = load_docs(model_path(model_name, test_path))
    print("Loaded {} test samples.".format(len(docs_test)))

    print("Building character dictioary...")
    chars_dict = build_chars_dict(docs_train)
    n_chars = len(chars_dict) + 1
    with io.open(model_path(model_name, CHARS_DICT_FILE), 'wb') as f:
        pkl.dump(chars_dict, f)
    print("{} different characters found.".format(n_chars - 1))

    print("Building label dictioary...")
    labels_dict = build_labels_dict(labels_train)
    n_classes = len(labels_dict) + 1
    with io.open(model_path(model_name, LABELS_DICT_FILE), 'wb') as f:
        pkl.dump(labels_dict, f)
    print("{} different labels found.".format(n_classes - 1))
    
    print("Vectorizing docs and labels...")
    x_chars_train, x_masks_train = vectorize_x(docs_train, chars_dict)
    x_chars_test, x_masks_test = vectorize_x(docs_test, chars_dict)
    y_train = vectorize_y(labels_train, labels_dict)
    y_test = vectorize_y(labels_test, labels_dict)
    
    model = Seyade(model_name)
    if args.load_model:
        print("Loading model from {}...".format(model.file_path(BEST_MODEL_FILE)))
        model.load()
    else:
        print("Building network...")
        model.build(n_chars, n_classes)
    
    print("Training...")

    batch_count = 0
    best_precision = 0.
    time_train_start = time.time()
    test_precisions = []
    
    try:
        for epoch in range(MAX_EPOCHS):
            
            print("Epoch {}".format(epoch))
            
            time_epoch_start = time.time()
            
            n_samples_epoch = 0
            train_cost_epoch = 0.
            
            train_result = {}
            train_result['docs'] = docs_train
            train_result['targets'] = y_train
            train_result['prediction_scores'] = []
            train_result['embeddings'] = []
            binary_prdedictions_train = []
            binary_targets_train = []
            
            # batch training
            x_chars_train, x_masks_train, y_train = shuffle_docs(x_chars_train, x_masks_train, y_train)
            for x_chars_batch, x_masks_batch, y_batch in batches(x_chars_train, x_masks_train, y_train):
                
                batch_count += 1
                n_samples_batch = len(x_chars_batch)
                n_samples_epoch += n_samples_batch
                
                train_cost_batch = model.train(x_chars_batch, x_masks_batch, y_batch)
                train_cost_epoch += train_cost_batch * n_samples_batch
                
                continuous = model.predict(x_chars_batch, x_masks_batch)
                # embeddings = model.embed(x_chars_batch, x_masks_batch)
                train_result['prediction_scores'].extend(continuous)
                # train_result['embeddings'].extend(embeddings)
                binary_prdedictions_train.extend(np.around(continuous).astype(int).tolist())
                binary_targets_train.extend(y_batch.tolist())
                
                if batch_count % DISPLAY_FREQUENCY == 0:
                    time_epoch_elapsed = time.time() - time_epoch_start
                    print("Epoch {} Batch {} Cost {:.8f} Time {}".format(epoch, batch_count, float(train_cost_batch), timedelta(seconds=time_epoch_elapsed)))
            
            model.save_result(train_result, TRAIN_RESULT_FILE)
            model.save_params()
            
            accuracy_train = jaccard_similarity_score(np.asarray(binary_targets_train), np.asarray(binary_prdedictions_train))
            precision_train = precision_score(np.asarray(binary_targets_train), np.asarray(binary_prdedictions_train), average='micro')
            recall_train = recall_score(np.asarray(binary_targets_train), np.asarray(binary_prdedictions_train), average='micro')
            
            print("Testing on test set...")
            test_result = {}
            test_result['docs'] = docs_test
            test_result['targets'] = y_test
            test_result['prediction_scores'] = []
            test_result['embeddings'] = []
            binary_prdedictions_test = []
            binary_targets_test = []
            
            for x_chars_batch, x_masks_batch, y_batch in batches(x_chars_test, x_masks_test, y_test):
                continuous = model.predict(x_chars_batch, x_masks_batch)
                # embeddings = model.embed(x_chars_batch, x_masks_batch)
                test_result['prediction_scores'].extend(continuous)
                # test_result['embeddings'].extend(embeddings)
                binary_prdedictions_test.extend(np.around(continuous).astype(int).tolist())
                binary_targets_test.extend(y_batch.tolist())
            
            model.save_result(test_result, TEST_RESULT_FILE)
            
            
            # Dispalay training summary
            accuracy_test = jaccard_similarity_score(np.asarray(binary_targets_test), np.asarray(binary_prdedictions_test))
            precision_test = precision_score(np.asarray(binary_targets_test), np.asarray(binary_prdedictions_test), average='micro')
            recall_test = recall_score(np.asarray(binary_targets_test), np.asarray(binary_prdedictions_test), average='micro')
            
            cost = train_cost_epoch / n_samples_epoch
            penalty = model.penalty()
            loss = cost - penalty
            
            print("Epoch {} Average Cost {:.8f} Loss {:.8f} Penalty {:.8f}".format(epoch, cost, loss, penalty))
            print("Epoch {} Training: accuracy {:.5f} precision {:.5f} recall {:.5f}".format(epoch, accuracy_train, precision_train, recall_train))
            print("Epoch {} Test:     accuracy {:.5f} precision {:.5f} recall {:.5f}".format(epoch, accuracy_test, precision_test, recall_test))
            print("Epoch {} Total Time {}".format(epoch, timedelta(seconds=time.time() - time_epoch_start)))
            
    except KeyboardInterrupt:
        pass
    
    model.save_params()
    print("Total Training Time = {}".format(timedelta(seconds=time.time() - time_train_start)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model_name', default='example', help='model dir name')
    parser.add_argument('--train_path', default='train.txt', help='train file name')
    parser.add_argument('--test_path', default='test.txt', help='test file name')
    parser.add_argument('--load_model', action="store_true", help='use trained model')
    args = parser.parse_args()
    main(args)

# python tools/detector.py nets/person_vs_background_vs_random/prod.prototxt models/person_vs_background_vs_random/person_vs_background_vs_random_lr_0.00001_iter_100000.caffemodel  --mean data/person_only_lmdb/person_vs_background_vs_random_color_mean.binaryproto --dir data/person/image_patches --list data/person/assign/person_vs_background_vs_random_test.txt --outfile p_v_b_v_r_batch_br_merge_results.txt


# python tools/detector.py nets/person_background_and_random/prod.prototxt models/person_background_and_random/person_background_and_random_lr_0.00001_iter_100000.caffemodel --mean data/person_only_lmdb/person_background_and_random_color_mean.binaryproto --dir data/person/image_patches --list data/person/assign/person_vs_background_vs_random_test.txt --outfile p_v_b_r_batch_results.txt


python tools/detector.py nets/person_background_and_random/prod.prototxt models/person_background_only/person_background_only_lr_0.00001_iter_100000.caffemodel --mean data/person_only_lmdb/person_background_only_color_mean.binaryproto --dir data/person/image_patches --list data/person/assign/person_vs_background_vs_random_test.txt --outfile p_v_b_batch_results.txt

# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC
# https://github.com/ofwallaart/HAABSA
# https://github.com/mtrusca/HAABSA_PLUS_PLUS

import nltk

import lcrModelAlt_hierarchical_v4
from HAABSA import lcrModelAlt_v4_fine_tune, lcrModelAlt_v4_test
from config import *
from loadData import *

nltk.download('punkt')


# Main function.
def main(_):
    # After running: back-up results file and model in case of running the model to be saved.
    # It is recommended to turn on logging of the output and to back that up as well.

    rest_lapt = False
    lapt_lapt = False
    book_book = False
    small_small = False
    rest_lapt_lapt = False
    rest_book_book = False
    rest_small_small = False
    rest_test = False
    write_result = True

    if rest_lapt:
        # Run and save restaurant-laptop.
        run_regular(source_domain="restaurant", target_domain="laptop", year=2014, learning_rate=0.001, keep_prob=0.7,
                    momentum=0.85, l2_reg=0.00001, write_result=write_result, savable=True)

    if lapt_lapt:
        # Run laptop-laptop for all splits.
        run_split(source_domain="laptop", target_domain="laptop", year=2014, splits=9, split_size=250,
                  learning_rate=0.001, keep_prob=0.7, momentum=0.85, l2_reg=0.00001, write_result=write_result)

    if book_book:
        # Run book-book for all splits.
        run_split(source_domain="book", target_domain="book", year=2019, splits=10, split_size=250,
                  learning_rate=0.001, keep_prob=0.7, momentum=0.85, l2_reg=0.00001, write_result=write_result)

    if small_small:
        # Hyper parameters (learning_rate, keep_prob, momentum, l2_reg).
        hyper_hotel = [0.001, 0.7, 0.85, 0.00001]
        hyper_apex = [0.001, 0.7, 0.85, 0.00001]
        hyper_camera = [0.001, 0.7, 0.85, 0.00001]
        hyper_creative = [0.001, 0.7, 0.85, 0.00001]
        hyper_nokia = [0.001, 0.7, 0.85, 0.00001]

        domains = [["hotel", 2014, hyper_hotel], ["Apex", 2004, hyper_apex], ["Camera", 2004, hyper_camera],
                   ["Creative", 2004, hyper_creative], ["Nokia", 2004, hyper_nokia]]

        # Run all single run models.
        for domain in domains:
            run_regular(source_domain=domain[0], target_domain=domain[0], year=domain[1], learning_rate=domain[2][0],
                        keep_prob=domain[2][1], momentum=domain[2][2], l2_reg=domain[2][3], write_result=write_result,
                        savable=False)

    if rest_lapt_lapt:
        # Run laptop-laptop fine-tuning on restaurant model.
        run_fine_tune(original_domain="restaurant", source_domain="laptop", target_domain="laptop", year=2014, splits=9,
                      split_size=250, learning_rate=0.02, keep_prob=0.6, momentum=0.85, l2_reg=0.00001,
                      write_result=write_result)

    if rest_book_book:
        # Run book-book fine-tuning on restaurant model.
        run_fine_tune(original_domain="restaurant", source_domain="book", target_domain="book", year=2014, splits=10,
                      split_size=250, learning_rate=0.001, keep_prob=0.6, momentum=0.99, l2_reg=0.001,
                      write_result=write_result)

    if rest_small_small:
        # Hyper parameters (learning_rate, keep_prob, momentum, l2_reg).
        hyper_hotel = [0.001, 0.7, 0.85, 0.00001]
        hyper_apex = [0.001, 0.7, 0.85, 0.00001]
        hyper_camera = [0.001, 0.7, 0.85, 0.00001]
        hyper_creative = [0.001, 0.7, 0.85, 0.00001]
        hyper_nokia = [0.001, 0.7, 0.85, 0.00001]

        # Run fine-tuning on restaurant model for all single run models.
        domains = [["hotel", 2014, hyper_hotel], ["Apex", 2004, hyper_apex], ["Camera", 2004, hyper_camera],
                   ["Creative", 2004, hyper_creative], ["Nokia", 2004, hyper_nokia]]

        for domain in domains:
            run_fine_tune(original_domain="restaurant", source_domain=domain[0], target_domain=domain[0],
                          year=domain[1], splits=0, split_size=0, learning_rate=domain[2][0], keep_prob=domain[2][1],
                          momentum=domain[2][2], l2_reg=domain[2][3], write_result=write_result, split=False)

    if rest_test:
        # Run restaurant test data through restaurant model.
        run_test(source_domain="restaurant", target_domain="restaurant", year=2014, write_result=write_result)

    print('Finished program successfully.')


# Run base model which can be saved for fine-tuning.
def run_regular(source_domain, target_domain, year, learning_rate, keep_prob, momentum, l2_reg, write_result, savable):
    # Hyper parameters.
    FLAGS.learning_rate = learning_rate
    FLAGS.keep_prob1 = keep_prob
    FLAGS.keep_prob2 = keep_prob
    FLAGS.momentum = momentum
    FLAGS.l2_reg = l2_reg

    # Other flags.
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.year = year
    FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    if write_result:
        with open(FLAGS.results_file, "w") as results:
            results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + "\n---\n")
        FLAGS.writable = 1

    # Run main method.
    if savable:
        FLAGS.savable = 1
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, False)
    _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, FLAGS.test_path, 1.0, test_size,
                                                                    test_size, FLAGS.learning_rate, FLAGS.keep_prob1,
                                                                    FLAGS.momentum, FLAGS.l2_reg)
    tf.reset_default_graph()
    FLAGS.savable = 0


# Runs LCR-Rot-hop++ for multiple training splits.
def run_split(source_domain, target_domain, year, splits, split_size, learning_rate, keep_prob, momentum, l2_reg,
              write_result):
    # Hyper parameters.
    FLAGS.learning_rate = learning_rate
    FLAGS.keep_prob1 = keep_prob
    FLAGS.keep_prob2 = keep_prob
    FLAGS.momentum = momentum
    FLAGS.l2_reg = l2_reg

    # Other flags.
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.year = year
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write("")
        FLAGS.writable = 1

    # Run main method.
    for i in range(1, splits + 1):
        print("Running " + FLAGS.source_domain + " to " + FLAGS.target_domain + " using " + str(
            split_size * i) + " aspects...")
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + " using " + str(
                    split_size * i) + " aspects\n---\n")
        FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT_" + str(
            split_size * i) + ".txt"
        train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, False)
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, FLAGS.test_path, 1.0,
                                                                        test_size, test_size, FLAGS.learning_rate,
                                                                        FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
        tf.reset_default_graph()


# Runs fine-tuning on a model originally trained on another domain to adapt for cross-domain use.
# Fine-tune method must be slightly adapted to work on original domains other than restaurant.
def run_fine_tune(original_domain, source_domain, target_domain, year, splits, split_size, learning_rate, keep_prob,
                  momentum, l2_reg, write_result, split=True):
    # Hyper parameters.
    FLAGS.learning_rate = learning_rate
    FLAGS.keep_prob1 = keep_prob
    FLAGS.keep_prob2 = keep_prob
    FLAGS.momentum = momentum
    FLAGS.l2_reg = l2_reg

    # Other flags.
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.year = year
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + original_domain + "_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write("")
        FLAGS.writable = 1

    if split:
        for i in range(1, splits + 1):
            print(
                "Running " + original_domain + " model with " + FLAGS.source_domain + " fine-tuning to " + FLAGS.target_domain + " using " + str(
                    split_size * i) + " aspects...")
            if FLAGS.writable == 1:
                with open(FLAGS.results_file, "a") as results:
                    results.write(
                        original_domain + " to " + FLAGS.target_domain + " with " + FLAGS.source_domain + " fine-tuning using " + str(
                            split_size * i) + " aspects\n---\n")
            FLAGS.train_path = "data/programGeneratedData/BERT/" + source_domain + "/" + str(
                FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT_" + str(
                split_size * i) + ".txt"
            train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, False)
            _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_v4_fine_tune.main(FLAGS.train_path, FLAGS.test_path, 1.0,
                                                                         test_size, test_size, FLAGS.learning_rate,
                                                                         FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
            tf.reset_default_graph()
    else:
        FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT.txt"
        print(
            "Running " + original_domain + " model with " + FLAGS.source_domain + " fine-tuning to " + FLAGS.target_domain + "...")
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(
                    original_domain + " to " + FLAGS.target_domain + " with " + FLAGS.source_domain + " fine-tuning\n---\n")
        train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, False)
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_v4_fine_tune.main(FLAGS.train_path, FLAGS.test_path, 1.0,
                                                                     test_size, test_size, FLAGS.learning_rate,
                                                                     FLAGS.keep_prob1, FLAGS.momentum, FLAGS.l2_reg)
        tf.reset_default_graph()


# Runs the test data through the model from the original domain.
def run_test(source_domain, target_domain, year, write_result):
    # Other flags.
    FLAGS.source_domain = source_domain
    FLAGS.target_domain = target_domain
    FLAGS.year = year
    FLAGS.train_path = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    if write_result:
        FLAGS.results_file = "data/programGeneratedData/" + str(
            FLAGS.embedding_dim) + "results_" + source_domain + "_" + FLAGS.target_domain + "_test_" + str(
            FLAGS.year) + ".txt"
        with open(FLAGS.results_file, "w") as results:
            results.write(source_domain + " to " + FLAGS.target_domain + "\n---\n")
        FLAGS.writable = 1

    # Run main method.
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, False)
    _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_v4_test.main(FLAGS.test_path, 1.0, test_size, test_size)
    tf.reset_default_graph()


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()

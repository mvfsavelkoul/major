# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC
# https://github.com/ofwallaart/HAABSA

import nltk

import cabascModel
import lcrModel
import lcrModelAlt
import lcrModelAlt_hierarchical_v4
import lcrModelInverse
import svmModel
from OntologyReasoner import OntReasoner
from loadData import *

nltk.download('punkt')

# import parameter configuration and data paths
from config import *

# import modules
import numpy as np


# Main function.
def main(_):
    altv4 = True
    rest_lapt = True
    lapt_lapt = True
    rest_lapt_lapt = False
    writeResult = True

    if altv4:
        FLAGS.train_path = "data/programGeneratedData/BERT/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT.txt"
        FLAGS.test_path = "data/programGeneratedData/BERT/" + str(
            FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"

        # Restaurant hyper parameters.
        FLAGS.learning_rate = 0.02
        FLAGS.keep_prob1 = 0.3
        FLAGS.keep_prob2 = 0.3
        FLAGS.momentum = 0.95
        FLAGS.l2_reg = 0.00001
        if rest_lapt:
            # Run restaurant-laptop.
            FLAGS.source_domain = "restaurant"
            FLAGS.target_domain = "laptop"
            if writeResult:
                with open(FLAGS.results_file, "w") as results:
                    results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + "\n---\n")
                FLAGS.writable = 1
            run_main()

        # Laptop hyper parameters.
        FLAGS.learning_rate = 0.02
        FLAGS.keep_prob1 = 0.3
        FLAGS.keep_prob2 = 0.3
        FLAGS.momentum = 0.95
        FLAGS.l2_reg = 0.00001
        if lapt_lapt:
            # Run laptop-laptop for different sizes.
            FLAGS.source_domain = "laptop"
            FLAGS.target_domain = "laptop"
            FLAGS.train_data = "data/externalData/" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + ".xml"
            FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
                FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
            if writeResult:
                FLAGS.results_file = "data/programGeneratedData/" + str(
                    FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
                    FLAGS.year) + ".txt"
                with open(FLAGS.results_file, "w") as results:
                    results.write("")
                FLAGS.writable = 1
            for i in range(1, 10):
                print("Running " + FLAGS.source_domain + " to " + FLAGS.target_domain + " for " + str(
                    250 * i) + " aspects...")
                if FLAGS.writable == 1:
                    with open(FLAGS.results_file, "a") as results:
                        results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + " for " + str(
                            250 * i) + " aspects\n---\n")
                FLAGS.train_path = "data/programGeneratedData/BERT/splits/" + str(
                    FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + "_BERT_" + str(
                    250 * i) + ".txt"
                run_main()

        if rest_lapt_lapt:
            # Run fine-tuned restaurant-laptop for different sizes.
            print()
        FLAGS.writable = 0


# Run main function.
def run_main():
    loadData = False
    useOntology = False
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    runLCRROTALT = False
    runLCRROTALTV4 = True
    runSVM = False

    weightanalysis = False

    # determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM:
        backup = True
    else:
        backup = False

    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)

    # print(test_size)
    # Was 250, 0.87
    remaining_size = test_size
    accuracyOnt = 1.00

    if useOntology == True:
        print('Starting Ontology Reasoner')
        Ontology = OntReasoner()
        # out of sample accuracy
        accuracyOnt, remaining_size = Ontology.run(backup, FLAGS.test_path, runSVM)
        # in sample accuracy
        # Ontology = OntReasoner()
        # accuracyInSampleOnt, remaining_size = Ontology.run(backup, FLAGS.train_path, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
        print('test acc={:.4f}, remaining size={}'.format(accuracyOnt, remaining_size))
        with open(FLAGS.results_file, "a") as results:
            if FLAGS.writable == 1:
                results.write("---\nOntology. Test accuracy: {:.4f}\n".format(accuracyOnt))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    # LCR-Rot model
    if runLCRROT == True:
        _, pred1, fw1, bw1, tl1, tr1, sent, target, true = lcrModel.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                         remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-inv model
    if runLCRROTINVERSE == True:
        lcrModelInverse.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-hop model
    if runLCRROTALT == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size,
                                                        FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.momentum,
                                                        FLAGS.l2_reg)
        tf.reset_default_graph()

    # LCR-Rot-Hop++
    if runLCRROTALTV4 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size,
                                                                        FLAGS.learning_rate, FLAGS.keep_prob1,
                                                                        FLAGS.momentum,
                                                                        FLAGS.l2_reg)
        tf.reset_default_graph()

    # CABASC model
    if runCABASC == True:
        _, pred3, weights = cabascModel.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size)
        if weightanalysis and runLCRROT and runLCRROTALT:
            outF = open('sentence_analysis.txt', "w")
            dif = np.subtract(pred3, pred1)
            for i, value in enumerate(pred3):
                if value == 1 and pred2[i] == 0:
                    sentleft, sentright = [], []
                    flag = True
                    for word in sent[i]:
                        if word == '$t$':
                            flag = False
                            continue
                        if flag:
                            sentleft.append(word)
                        else:
                            sentright.append(word)
                    print(i)
                    outF.write(str(i))
                    outF.write("\n")
                    outF.write(
                        'lcr pred: {}; CABASC pred: {}; lcralt pred: {}; true: {}'.format(pred1[i], pred3[i], pred2[i],
                                                                                          true[i]))
                    outF.write("\n")
                    outF.write(";".join(sentleft))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in fw1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentright))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in bw1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(target[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tl1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tr1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentleft))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in fw2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentright))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in bw2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(target[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tl2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tr2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sent[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in weights[i][0]))
                    outF.write("\n")
            outF.close()

    # BoW model
    if runSVM == True:
        svmModel.main(FLAGS.train_svm_path, test, accuracyOnt, test_size, remaining_size)

    print('Finished program successfully.')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()

# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC
# https://github.com/ofwallaart/HAABSA

import tensorflow as tf
import lcrModel
import lcrModelAlt_hierarchical_v4
import lcrModelInverse
import lcrModelAlt
import cabascModel
import svmModel
from OntologyReasoner import OntReasoner
from loadData import *
import nltk
import xml.etree.ElementTree as ET
import os

nltk.download('punkt')

# import parameter configuration and data paths
from config import *

# import modules
import numpy as np
import sys

# Main function.
def main(_):
    splitData = True
    lcrRotHop = True
    altv4 = True
    writeResult = True

    if splitData:
        split_data()

    if lcrRotHop:
        if writeResult:
            with open(FLAGS.results_file, "w") as results:
                results.write("")
            FLAGS.writable = 1
        for i in range(10, 101, round(100/FLAGS.splits)):
            print("Running " + FLAGS.source_domain + " to " + FLAGS.target_domain + " at " + str(i) + "%...")
            if FLAGS.writable == 1:
                with open(FLAGS.results_file, "a") as results:
                    results.write(FLAGS.source_domain + " to " + FLAGS.target_domain + " at " + str(i) + "%\n---\n")
            run_main(i, altv4)
    FLAGS.writable = 0
# Split data.
def split_data():
    # Load training data.
    train_tree = ET.parse(FLAGS.train_data)
    train_root = train_tree.getroot()
    train_sentences = train_root.findall("sentence")

    # Load testing data.
    # test_tree = ET.parse(FLAGS.test_data)
    # test_root = test_tree.getroot()
    # test_sentences = test_root.findall("sentence")

    for i in range(10, 101, round(100/FLAGS.splits)):
        # Split train data.
        numSen = round((i/100) * len(train_sentences))
        sentences = ET.Element('sentences')
        for j in range(numSen):
            sentence = train_sentences[j]
            sentences.append(sentence)
        sentencesXML = ET.tostring(sentences, encoding="unicode")
        with open("data/programGeneratedData/splits/"+FLAGS.source_domain+"_train_"+str(FLAGS.year)+"_"+str(i)+".xml", "w") as tSplit:
            tSplit.write(sentencesXML)

        # Split test data
        # numSen = round((i/100) * len(test_sentences))
        # sentences = ET.Element('sentences')
        # for j in range(numSen):
        #    sentence = test_sentences[j]
        #    sentences.append(sentence)
        # sentencesXML = ET.tostring(sentences, encoding="unicode")
        # with open("data/programGeneratedData/splits/"+FLAGS.target_domain+"_test_"+str(FLAGS.year)+"_"+str(i)+".xml", "w") as tSplit:
        #    tSplit.write(sentencesXML)

# Run main function.
def run_main(splitpercent, altv4):
    loadData = True
    useOntology = False
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    runLCRROTALT = True
    runLCRROTALTV4 = False
    runSVM = False

    if altv4:
        runLCRROTALT = False
        runLCRROTALTV4 = True

    weightanalysis = False

    # determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM:
        backup = True
    else:
        backup = False

    # retrieve data and wordembeddings
    FLAGS.train_data = "data/programGeneratedData/splits/"+FLAGS.source_domain+"_train_"+str(FLAGS.year)+"_"+str(splitpercent)+".xml"
    # FLAGS.test_data = "data/programGeneratedData/splits/"+FLAGS.target_domain+"_test_"+str(FLAGS.year)+"_"+str(splitpercent)+".xml"
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)

    #print(test_size)
    # Why?
    remaining_size = 250
    accuracyOnt = 0.87

    if useOntology == True:
        print('Starting Ontology Reasoner')
        Ontology = OntReasoner()
        # out of sample accuracy
        accuracyOnt, remaining_size = Ontology.run(backup, FLAGS.test_path, runSVM)
        # in sample accuracy
        #Ontology = OntReasoner()
        #accuracyInSampleOnt, remaining_size = Ontology.run(backup, FLAGS.train_path, runSVM)
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
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path, test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-Hop++
    if runLCRROTALTV4 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
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
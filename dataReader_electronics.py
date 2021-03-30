import nltk

nltk.download('punkt')


def main():
    merge = False

    in_path = "data/externalData/electronics_reviews_2004/"
    files = ["Apex AD2600 Progressive-scan DVD player.txt", "Camera.txt",
             "Creative Labs Nomad Jukebox Zen Xtra 40GB.txt", "Nokia 6610.txt"]
    out_path = "data/programGeneratedData/BERT/electronics/raw_data_electronics_2004.txt"

    if merge:
        with open(out_path, "w") as out:
            for file in files:
                lines = open(in_path + file).readlines()
                for line in lines[12:]:
                    if line[0:3] == "[t]" or line[0:2] == "##":
                        continue
                    else:
                        split = line.split("##")
                        line = split[1]
                        aspects = split[0].split(",")
                        for aspect in aspects:
                            words = nltk.word_tokenize(aspect)
                            if 'u' in words or 'p' in words or 's' in words or 'cc' in words or 'cs' in words:
                                continue
                            else:
                                count += 1
                                aspect = words[0]
                                for word in words[1:]:
                                    if word == "-3" or word == "-2" or word == "-1" or word == "+1" or word == "+2" or word == "+3":
                                        polarity = int(int(word) / abs(int(word)))
                                    elif word == '[' or word == ']':
                                        continue
                                    else:
                                        aspect += " " + word
                                out.write(line.replace(aspect, "$T$"))
                                out.write(aspect + "\n")
                                out.write(str(polarity) + "\n")
        print("Read " + count + " aspects.")
    else:
        for file in files:
            count = 0
            name_split = file.split()
            if len(name_split) == 1:
                name = name_split[0].split('.')[0]
            else:
                name = name_split[0]
            out_path = "data/programGeneratedData/BERT/" + name + "/raw_data_" + name + "_2004.txt"
            with open(out_path, "w") as out:
                lines = open(in_path + file).readlines()
                for line in lines[12:]:
                    if line[0:3] == "[t]" or line[0:2] == "##":
                        continue
                    else:
                        split = line.split("##")
                        line = split[1]
                        aspects = split[0].split(",")
                        for aspect in aspects:
                            words = nltk.word_tokenize(aspect)
                            if 'u' in words or 'p' in words or 's' in words or 'cc' in words or 'cs' in words:
                                continue
                            else:
                                count += 1
                                aspect = words[0]
                                for word in words[1:]:
                                    if word == "-3" or word == "-2" or word == "-1" or word == "+1" or word == "+2" or word == "+3":
                                        polarity = int(int(word) / abs(int(word)))
                                    elif word == '[' or word == ']':
                                        continue
                                    else:
                                        aspect += " " + word
                                out.write(line.replace(aspect, "$T$"))
                                out.write(aspect + "\n")
                                out.write(str(polarity) + "\n")
            print("Read " + str(count) + " aspects from " + name)


if __name__ == '__main__':
    main()

import random


def good(data):
    i = 0
    with open('pos.txt', 'w+') as writer:
        with open(data, 'r+') as f:
            for line in f:
                i += 1
                with open('/Users/mariiaaksenova/PycharmProjects/python-autofill/' + line[:-1], 'r+') as text:
                    try:
                        for line in text:
                            mas = line.split()
                            if len(mas) == 0 or mas[0] == '#' or mas[0] == '#--':
                                pass
                            else:
                                for k in range(len(mas) - 1):
                                    writer.writelines(str(mas[k]) + ' ' + str(mas[k + 1]) + '\n')
                        if i == 25:
                            break
                    except Exception as e:
                        print(e)


def bad(data):
    i = 0
    with open('neg.txt', 'w+') as writer:
        with open(data, 'r+') as f:
            for line in f:
                i += 1
                with open('/Users/mariiaaksenova/PycharmProjects/python-autofill/' + line[:-1], 'r+') as text:
                    try:
                        for line in text:
                            mas = line.split()
                            if len(mas) == 2:
                                writer.writelines(str(mas[1]) + ' ' + str(mas[0]) + '\n')
                            elif len(mas) == 0 or mas[0] == '#' or mas[0] == '#--':
                                pass
                            else:
                                for k in range(len(mas)):
                                    n = random.randint(0, len(mas) - 1)
                                    if n != k + 1:
                                        writer.writelines(str(mas[k]) + ' ' + str(mas[n]) + '\n')
                                    else:
                                        try:
                                            writer.writelines(str(mas[k]) + ' ' + str(mas[n + 1]) + '\n')
                                        except:
                                            writer.writelines(str(mas[k]) + ' ' + str(mas[n - 1]) + '\n')

                        if i == 25:
                            break
                    except Exception as e:
                        print(e)


def union(file1,file2):
    with open('union.txt','w+') as writer:
        with open(file1,'r+') as f1:
            for line in f1:
                writer.writelines(line)
        with open(file2,'r+') as f2:
            for line in f2:
                writer.writelines(line)

if __name__ == '__main__':
    good('python100k_train.txt')
    bad('python100k_train.txt')
    union('pos.txt','neg.txt')

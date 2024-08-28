import chardet

with open('data.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))
    print(result)

# Lisa Pegram

#Assume that a file containing a series of names (as strings) is named names.txt and exists on the computer’s disk. Write a program that displays the number of names that are stored in the file.


def main():
    # assign values to 0 count can never be float
    total = 0.0

    # open file with numbers to be averaged
    pathnameIn = r'C:\Users\Lisa.Pegram\Desktop\SQL'
    filenameIn = 'names.txt'
    pathIn = pathnameIn + "/" + filenameIn

    names = open(pathIn, 'r')

    #with open(pathIn, 'r') as file_:
    #    names= file_.read().replace('\n',' ')

    print(names)

    # iterate for loop for all lines in 'file'
    for i in names:
       total+=1

#print the number of things you have to get right
    print("the number of names in this file is", total)

main()

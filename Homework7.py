# Lisa Pegram

# Assume that a file containing a series of integers is named numbers.txt
# and exists on the computer's disk. Write a program that calulates the
# average of all the numbers stored in the file.

# for the sake of testing you can use exercise 7 to generate
# a file that will provide numbers to average

def main():
    # assign values to 0 count can never be float
    total = 0.0

    # open file with numbers to be averaged
    pathnameIn = r'C:\Users\Lisa.Pegram\Desktop\SQL'
    filenameIn = 'correct_answers.txt'
    pathIn = pathnameIn + "/" + filenameIn

    # open file with numbers to be averaged
    pathnameIn2 = r'C:\Users\Lisa.Pegram\Desktop\SQL'
    filenameIn2 = 'student_answers.txt'
    pathIn2 = pathnameIn2 + "/" + filenameIn2

    pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\SQL'
    filenameOut = 'graded.txt'
    pathOut = pathnameOut + "/" + filenameOut

    with open(pathIn, 'r') as file_:
        correct= file_.read().replace('\n','')
    with open(pathIn2, 'r') as file_2:
        student= file_2.read().replace('\n','')

    # iterate for loop for all lines in 'file'
        output = []
        for i,i2 in zip(correct,student):
            if i == i2:
                output.append('Right')
                total+=1
                #print(i, i2,output)
                #print("The number of problems wrong 'student_answers.txt' is:", output)
            else:
                output.append('Wrong')
                #print(i, i2, output)
                #print("The number of problems wrong 'student_answers.txt' is:", output)
        print("the score is", total,"out of 20, earning the student a ", total*5, '% grade')
        print("The number of problems wrong 'student_answers.txt' is:", output)

    fileOut = open(pathOut, 'w')
    fileOut.write(str(total))



main()
#Programming Exercise 12

def main():
 # Local variable
 number = 0

 # Get number as input from the user.
# number = int(input('How many numbers to display? '))
 # number2 = int(input('How many numbers to display? '))
 #Design a recursive function that accepts two arguments into the parameters  x and y.The function should return the value of x times y.Remember, multiplication can be performed as repeated addition as follows:
 #print_num(number, number2)
 factorial(2,3)
 # Display the numbers.
 #print_num(number,number2)


# The print_num function is a a recursive function
# that accepts an integer argument, n , and prints
# the numbers 1 up through n.
def print_num(n,j):
    total=0
    while n>1:
     print_num(n-1,j)
             #print (total, sep=' ')
    print(j)
    #print(total)
    #print(n)

def factorial( n,j ):
    print(j)
    if j<=1:   # base case
       return 1
    else:
       return n + factorial(n,j-1)  # recursive call
       #print(str(n) + '= round : ')
       #return returnNumber

#x=0
#y=0

#get x and y
#x=int(input('what is x?'))
#y=int(input('what is y?'))

#def multiplication(x,y):
# i=0
# if x>0 and y>0:
#     for i in (i,y):
#        x+=x
# #       i+=1
#         print(x)
# else:
#     print('error')


# Call the main function.
main()

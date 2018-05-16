def RecursiveProduct(a, b):
      # Base case 1
    if a == 0 or b == 0:
        return 0

        # Base case 2
    if b == 1:
        return a

    return a + RecursiveProduct(a, b - 1)

def main():
    Finalanswer=RecursiveProduct(3, 4)
    print("3 x 4 = {",Finalanswer,"}")

main()




##Input the number of fat grams
fat_grams=int(input("What is the number of fat grams that you consume in a day? "))

##Input the number of carb grams
carb_grams=int(input("What is the number of carb grams that you consume in a day? "))

#calories from fat=fat grams×9
calories_from_fat= fat_grams*9
#calories from carbs=carb grams×4
calories_from_carbs=carb_grams*4

print("Your calories from fat are:",calories_from_fat)
print("Your calories from carbs are:",calories_from_carbs)
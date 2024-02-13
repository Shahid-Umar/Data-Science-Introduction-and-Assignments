# Calculating BMI (Body Mass Index)
# Taking input from user

name = input("Please enter your name : ")
weight = float(input("Please enter your weight (in kg) : "))
height = float(input("Please enter your height (in meters) : "))

# Calculate BMI
bmi = round((weight/height**2),2)

# Condition for checking criteria
if bmi <= 18.5:
    print(name, "You are underweight.")
elif 18.5 < bmi <= 24.9:
    print(name, "Your weight is normal.")
elif 25 < bmi <= 29.29:
    print(name, "You are overweight.")
else:
    print(name, "You are obese.")

# Christian Quiroz
# MIS 649 - Module 4 Assignment

auto_raw = read.csv("https://www.statlearning.com/s/Auto.csv", header=T, na.string="?") 

auto = na.omit(auto_raw)

attach(auto)

#====================================================================================================================

# Question 1: 

origin = as.factor(origin)
cylinders = as.factor(cylinders)

model1 = lm(mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin)
summary(model1)


#====================================================================================================================

# Question 2:

par(mfrow=c(2,2))
plot(model1)


#====================================================================================================================

# Question 3: 

# Applying log to weight. 
model2 = lm(mpg ~ cylinders + displacement + horsepower + log(weight) + acceleration + year + origin)
summary(model2)

par(mfrow=c(2,2))
plot(model2)

# Applying sqrt to displacement and log to weight. 
model3 = lm(mpg ~ cylinders + sqrt(displacement) + horsepower + log(weight) + acceleration + year + origin)
summary(model3)

par(mfrow=c(2,2))
plot(model3)


#====================================================================================================================

# Question 4:

# Applying interaction terms
model4 = lm(mpg ~ cylinders:horsepower + displacement + weight:acceleration + year + origin)
summary(model4)

par(mfrow=c(2,2))
plot(model4)


#====================================================================================================================

# Question 5:

test_data = data.frame(cylinders=factor(6), displacement=300, horsepower=130, weight=3204, acceleration=12.0, year=80, origin=factor(3))

predict_mpg = predict(model3, newdata=test_data)

predict_mpg

detach(auto)


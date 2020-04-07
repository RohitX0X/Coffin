using CSV
using Plots
using Statistics
using LinearAlgebra
include("fuunctions.jl")
# cd("D:\\dload\\iitb\\4th sem\\GNR_652\\assgnments\\180100010")
# Y = β_0 * X_0 + β_1 * X_1 + β_2 * X_2
# if X_0 = 1: we can write the above as Y = β * X for vectorization
# Predictions from Model β: Y_Pred = β * X
# pwd()
# Read the dataset from file
dataset = CSV.read("data/housingPriceData.csv")
x1_tot = dataset[:,3]
x2_tot = dataset[:,4]
x3_tot = dataset[:,5]
x1=dataset[1:17291,3]
# x2=dataset[1:17291,4]
# x3=dataset[1:17291,5]
# x1_test=x1_tot[17292:21614,3]
# x2_test=x2_tot[17292:21614,4]
# x3_test=x3_tot[17292:21614,5]
m_tot = length(x1_tot)
# print(m_tot)
x0_tot = ones(m_tot)
X_tot = cat(x0_tot, x1_tot, x2_tot,x3_tot, dims=2)
X_tot=normalized(X_tot,m_tot)
# X_tot[:,1]=ones(m_tot)
X=X_tot[1:17291,:]
# print(X.shape)
X_test=X_tot[17292:21613,:]




# # Visualize the columns as a scatter plot
# scatter3d(course1, course2, course3)
#
# # Stub column 1 for vectorization.
m = 17291
# print(m)
# x0 = ones(m)
# X = cat(x0, x1, x2,x3, dims=2)
# X=normalized(X,m)

# Get the variable we want to regress
Y=zeros(m,1)
S= dataset[1:17291,2]
Y=S+Y
m_test=length(x1_tot)-length(x1)
Ytest=zeros(m_test,1)
Stest=dataset[17292:21613,2]
Ytest=Stest+Ytest


Ymean=sum(Ytest)/m_test
# Y=normalized(Y,m)
# # Initial coefficients
B = zeros(4, 1)
# Calcuate the cost with intial model parameters B=[0,0,0]
intialCost = costFunction(X, Y, B)

learningRate = 100
newB, costHistory = gradientDescent(X, Y, B, learningRate, 15000)
# print(newB)

# Make predictions using the learned model; newB
YPred = X * newB
YPred_test=X_test*newB
# print(YPred_test)
# CSV.write("data/1a.csv",YPred_test,append=true)
rmss=RMSS(X_test,Ytest,newB)
r2=R2(X_test,Ytest,newB,Ymean)
# print("the RMSS score for linear regression is",":",rmss,"  ","the R2 score is",":",r2)

# visualize and compare the the prediction with original; below we plot only first 10 entries; plot! is to plot on the existing plot window
# plot(Y[1:100])
# plot!(YPred[1:100])
#
# # Visualize the learning: how the loss decreased.
# plot(costHistory)
using DataFrames
Price = X_test*newB
df = DataFrame(Price)
CSV.write("data\\1a.csv", df)

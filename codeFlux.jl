using Statistics, Flux

inputs = [.2, -.3, .5, 1, -.9]
outputs = [-.2, .3, -.5, -1, .9]
weigth = .4

# let us create a function to run a prediction
predict(inputs, weight) = inputs .* weight

# Now let us create the loss function
loss(inputs, outputs, weigth) = mean((outputs - (inputs .* weigth)) .^2)

# Finally, let us define the derivative of the loss function
dloss(inputs, outputs, weigth) = data(gradient(loss,inputs,outputs,weigth)[3])

for i in 1:200
    println("Current prediction: $(predict(inputs, weigth))")
    println("Current loss: $(loss(inputs, outputs, weigth))")
    println("Current weight: $(weigth)")
    d = dloss(inputs, outputs, weigth)
    global weigth -= d *.1
end

println("This is an implementation of perceptor neural network")

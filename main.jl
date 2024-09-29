using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using Flux
using Random
includet("functions.jl")

# main()

n_time_points = 4
p = 3
embedding_size = 3


aaa = 1.0
bbb = 0.0
ccc = 0.0
ddd = 1.0

                        input =                 
                        [aaa, 0.0, bbb,    # obs 1
                         0.0, 1.0, 0.0, 
                         0.0, 0.0, 0.0, 
                         ccc, 0.0, ddd,
                         aaa, 0.0, bbb,    # obs 2
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         ccc, 0.0, ddd,
                         0.0, 0.0, 0.0,    # obs 3
                         aaa, 0.0, bbb,
                         0.0, 1.0, 0.0,
                         ccc, 0.0, ddd,
                         aaa, 0.0, bbb,    # obs 4
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         aaa, 1.0, bbb,    # obs 5
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         ccc, 0.0, ddd,
                         0.0, 0.0, 0.0,    # obs 6
                         aaa, 0.0, bbb,
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0,    # obs 7
                         0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 1.0, 0.0]

startvec = [0, 4, 8, 12, 16, 20, 24]



# Random.seed!(11)

# function main()

epochs = 10000
learning_rate = 0.005


# relu is to make everything positive

query = Query(Dense(embedding_size, 1, relu))
# reinitialize all the weights using rand() * 0.2 .- 0.1
query.dense.weight .= reshape(rand(length(query.dense.weight)) * 0.2 .- 0.1, size(query.dense.weight)...)
query.dense.bias .= reshape(rand(length(query.dense.bias)) * 0.2 .- 0.1, size(query.dense.bias)...)

key = Key(Dense(embedding_size, 1, relu))
# reinitialize all the weights using rand() * 0.2 .- 0.1
key.dense.weight .= reshape(rand(length(key.dense.weight)) * 0.2 .- 0.1, size(key.dense.weight)...)
key.dense.bias .= reshape(rand(length(key.dense.bias)) * 0.2 .- 0.1, size(key.dense.bias)...)

value = Value(Dense(embedding_size, 1, relu))
# reinitialize all the weights using rand() * 0.2 .- 0.1
value.dense.weight .= reshape(rand(length(value.dense.weight)) * 0.2 .- 0.1, size(value.dense.weight)...)
value.dense.bias .= reshape(rand(length(value.dense.bias)) * 0.2 .- 0.1, size(value.dense.bias)...)


predictor = Predictor(Dense(1, embedding_size))
predictor.dense.weight .= reshape(rand(length(predictor.dense.weight)) * 0.2 .- 0.1, size(predictor.dense.weight)...)
predictor.dense.bias .= reshape(rand(length(predictor.dense.bias)) * 0.2 .- 0.1, size(predictor.dense.bias)...)

# print parameters
println("Query before training")
println(query.dense.weight)
println(query.dense.bias)


model = Chain(query, key, value, predictor)

sequences = prepare_sequences(input, startvec, n_time_points, p)

model = train_model!(model, sequences, learning_rate, epochs)

println("Query after training")
println(model[1].dense.weight)
println(model[1].dense.bias)

println("Key after training")
println(model[2].dense.weight)
println(model[2].dense.bias)


println("Value after training")
println(model[3].dense.weight)
println(model[3].dense.bias)

println("Predictor after training")
println(model[4].dense.weight)
println(model[4].dense.bias)

# end


# main()

using Statistics: mean
using Zygote

function prepare_sequences(input, startvec, n_time_points, p)
    sequences = []
    for j in 1:length(startvec)
        if j < length(startvec)
            curn = startvec[j + 1] - startvec[j]
        else
            curn = div(length(input), p) - (startvec[j])
        end
        curstart = startvec[j] * p + 1

        # for k = 0:curn-3
        #     push!(sequences, reshape(input[(curstart + k*p):(curstart + k*p + (curn-k)* p) - 1], curn-k, p))
        # end

        push!(sequences, reshape(input[curstart:(curstart + (n_time_points* p) - 1)], p, n_time_points)')

    end

    return sequences
end


struct Query
    dense::Dense
end

function (q::Query)(x) 
    return q.dense.σ.(x * q.dense.weight'  .+ q.dense.bias) .+ q.dense.bias
end

struct Key
    dense::Dense
end

function (k::Key)(x) 
    return k.dense.σ.(x * (k.dense.weight') .+ k.dense.bias) .+ k.dense.bias
end

struct Value
    dense::Dense
end

function (v::Value)(x) 
    return v.dense.σ.(x * (v.dense.weight') .+ v.dense.bias) .+ v.dense.bias
end

struct Predictor
    dense::Dense
end

function (p::Predictor)(x) 
    # not fully connected
    return p.dense.σ.(x .* p.dense.weight .+ p.dense.bias)
end

function loss(x, model)
    n_time_points, p = size(x)

    rpe = abs.(collect(1:n_time_points-1) .- collect(1:n_time_points-1)') .+ 1

    queryvec = model[1](x[1:n_time_points - 1, :])

    keyvec = model[2](x[1:n_time_points - 1, :])

    valuevec = model[3](x[1:n_time_points - 1, :])

    attn_matrix =softmax(((queryvec * keyvec') ./ rpe), dims = 2)

    # @show attn_matrix

    # encoding = sum(attn_matrix * valuevec)

    # pred = [model[4][i](encoding)[1] for i in 1:p]

    encoding = sum(attn_matrix * valuevec)

    pred = model[4](encoding)
    # pred = encoding

    return sum((x[n_time_points, :] .- pred).^2)
end

function get_params(model)

    params = []
    for obj in model
        try
            push!(params, obj.dense)
        catch 
            for element in obj
                push!(params, element.dense)
            end
        end
    end

    return params
end

function train_model!(model, sequences, learning_rate, epochs)
    # opt = ADAM(learning_rate)
    opt = Descent(learning_rate)
    ps = Flux.params(get_params(model)...)
    
    for epochs = 1:epochs
        @info "Epoch $epochs"

        for i = 1:length(sequences)
            gs = gradient(ps) do
                loss_val = loss(sequences[i], model)
            end
            Flux.Optimise.update!(opt, ps, gs)
            
        end

        println("Loss: ", mean([loss(sequences[i], model) for i in 1:length(sequences)]))
    end
           
    return model

end

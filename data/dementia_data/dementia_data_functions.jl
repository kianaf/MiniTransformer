using Statistics        
using Distributions: Chisq, ccdf
using LinearAlgebra

# Define threshold rules for impairment (you can adjust these based on the actual thresholds)
function impairment_classification(original_df)
    

    df = deepcopy(original_df)


    # MMSE (mms): Below 24 indicates cognitive impairment
    df[!, :mms] = ifelse.(df.mms .>= 24, 0, 1)

    # Isaac Set Test (isa_15): Below 30 indicates cognitive impairment
    df[!, :isa_15] = ifelse.(df.isa_15 .>= 30, 0, 1)

    # Subscales of Isaac’s Set Test (ISA-15)
    df[!, :cou_15] = ifelse.(df.cou_15 .>= 8, 0, 1)  # Colors: Below 8 is impaired
    df[!, :ani_15] = ifelse.(df.ani_15 .>= 10, 0, 1) # Animals: Below 10 is impaired
    df[!, :fru_15] = ifelse.(df.fru_15 .>= 8, 0, 1)  # Fruits: Below 8 is impaired
    df[!, :vil_15] = ifelse.(df.vil_15 .>= 7, 0, 1)  # Villages: Below 7 is impaired


    # Benton (benton): Below certain thresholds, e.g., <7 indicates impairment
    df[!, :benton] = ifelse.(df.benton .>= 7, 0, 1)

    # DSST (cod_w): Below 40 could indicate impairment
    df[!, :cod_w] = ifelse.(df.cod_w .>= 40, 0, 1)

    # CES-D score (csd): Threshold of 16 or higher indicates depression
    df[!, :csd] = ifelse.(df.csd .< 16, 0, 1)

    # Example CES-D item csd_14 ("I felt lonely") where 3 = high frequency of loneliness
    # FIXME: not sure which threshold makes more sense.
    df[!, :csd_14] = ifelse.(df.csd_14 .== 3, 1, 0)


    # "Hier" dependency: Define threshold for impairment, e.g., 0 or 1 is normal, 2 or higher is impaired
    # FIXME: not sure which threshold makes more sense.
    df[!, :hier] = ifelse.(df.hier .<= 1, 0, 1)

    # about dem, we only have it as one if the person is diagnosed with dementia now ro before. 
    #FIXME is it the right way to do it?

    # Define the target variable: 1 if the person has dementia, 0 otherwise



    return df
end



function correct_dem_variable!(df)

    # Sort the df by id and the visit number or time point
    sort!(df, [:id, :visit])  # Replace :visit with the actual column name for time or visit
    
    number_of_rows = size(df, 1)

    # Check if there's dem in any point of time for the individual, make the dem in next time points 1
    for i in 1:number_of_rows
        if df[i, :dem] == 1
            if i < number_of_rows
                
                j = i + 1
                while df[i, :id] == df[i+1, :id]
                    df[j, :dem] = 1
                    j = j + 1
                    i = i + 1
                end
            end
        end
    end


    return df
end


function medication_report(df_classified)

    # change the data to have two variables for medication both zero if no change in medication and 1 for one of them if there is an increase and 1 for the other if there is a decrease
    # if there is an increase in medication, the variable for increase will be 1 and the variable for decrease will be 0
    # if there is a decrease in medication, the variable for increase will be 0 and the variable for decrease will be 1

    # Sort the df by id and the visit number or time point
    sort!(df_classified, [:id, :visit])  # Replace :visit with the actual column name for time or visit

    # add two variables medication_increase and medication_decrease to the data frame
    df_classified[!, :medication_increase] .= 0
    df_classified[!, :medication_decrease] .= 0

    number_of_rows = size(df_classified, 1)


    for i = 2:number_of_rows
        if i < number_of_rows
            if df_classified[i, :id] == df_classified[i-1, :id]

                if df_classified[i, :medication] < df_classified[i-1, :medication]
                    df_classified[i, :medication_decrease] = 1
                elseif df_classified[i, :medication] > df_classified[i-1, :medication]
                    df_classified[i, :medication_increase] = 1
                end
            end
        end
    end

    # remove the medication variable from the data frame
    select!(df_classified, Not(:medication))

    return df_classified

end



function get_data_dir(save_path)
    
    data_number = get_last_data_dir_number(save_path)

    current_data_dir = string("$(pwd())/$(save_path)/data_$(data_number)")
    
    return current_data_dir

end


function get_last_data_dir_number(save_path)

    # @show save_path
    # !isdir(string(pwd())) && mkdir(string(pwd(), save_path))



    parent_dir = string(pwd(), save_path)
    dir_list = readdir(parent_dir)

    
    runs_list = []

    if length(dir_list) > 0
        for i = 1:length(dir_list)
            
            if isdir(string(parent_dir, "/",  dir_list[i])) && startswith(dir_list[i], "data_")                    # checking, if it's directory
                append!(runs_list, parse(Int, string(dir_list[i])[6:end]))      # print the name of a directory
            end
        end
        data_number = (sort(runs_list))[end]
    else
        data_number = 1
    end


    return data_number
end


function get_dementia_data()

    return CSV.read("../../../dementia_data/forStratos_T0T30_3772subjects_classified.csv", DataFrame)
end



function get_dictionary_plot_id_datapoint(df)

    ids = Set(df.id)

    frequency_id_dict = Dict()

    for i = 1: size(df,1)

        if haskey(frequency_id_dict, df.id[i])
            frequency_id_dict[df.id[i]] += 1
        else
            frequency_id_dict[df.id[i]] = 1
        end

    end

    median_frequency = Statistics.median(collect(values(frequency_id_dict)))

    median = Statistics.median(map(x->frequency_id_dict[x], collect(ids)))
    average = mean(map(x->frequency_id_dict[x], collect(ids)))

    return frequency_id_dict, length(ids),  median, average

end

function prepare_sequences_for_dementia_data()

    current_data_path = get_data_dir("/data_version")

    mkdir(current_data_path)

    df = get_dementia_data()

    included_events = ["id", "visit", "time", "dem", "mms", "cou_15", "ani_15", "fru_15", "vil_15", "isa_15", "benton", "cod_w", "csd", "csd_14", "livalone", "hier", "medication_increase", "medication_decrease"]

    df = df[:, included_events]


    # df, df_copy = denoising(df, args.min_streak_size)

    # chi_square_metric_matrix = get_chi_square_metric(df, "median")
    chi_square_metric_matrix = get_chi_square_metric(df) # default is method = count

   events_list = included_events[4:end]

    y_labels = reverse(events_list)
    x_labels = events_list


    heatmap(x_labels, y_labels, reverse(chi_square_metric_matrix, dims = 1), xticks = :all, yticks = :all, size =(1000, 700), xrotation = 90)
    savefig("$(current_data_path)/chi_square_metric_matrix.pdf")



    data_dict = make_dict_of_df(df)

    # data_dict = remove_empty_last_rows(data_dict)


    data_sentence_string_set = create_word_sequence_stressors(df, events_list, data_dict)


    data_sentence_string_set = remove_empty_sequences(data_sentence_string_set)


    groups = []

    for key in keys(data_sentence_string_set)
        for group in data_sentence_string_set[key]
            push!(groups, group)
        end
    end
    group_frequency_dict = Dict()

    for group in groups

        if haskey(group_frequency_dict, group)
            group_frequency_dict[group] += 1
        else
            group_frequency_dict[group] = 1
        end
    end
    

    isfile("$(current_data_path)/group_frequency_dict_dh_le.txt") && rm("$(current_data_path)/group_frequency_dict_dh_le.txt")
    log_dictionary(group_frequency_dict, "value", "$(current_data_path)/group_frequency_dict_dh_le.txt")


    threshold =  ceil(1 * length(collect(Set(df.id))))
    data_sentence_string_set_new = regroup_based_on_chi_square(data_sentence_string_set, chi_square_metric_matrix, threshold, included_events[4:end])

    

    # if args.appearance_disappearance_flag
    #     new_included_stressors = []
    #     data_sentence_string_set_new = change_sequence_to_appearance_disappearance!(data_sentence_string_set_new)
        
    #     for stressor in args.included_stressors
            
    #         if contains(stressor, "ghq")
    #             push!(new_included_stressors, stressor)
    #         else
    #             push!(new_included_stressors, "appearance_$(stressor)")
    #             push!(new_included_stressors, "disappearance_$(stressor)")
    #         end
    #     end

    #     args.included_stressors = new_included_stressors
    #     println(args.included_stressors)
    # end

    regroup_frequency_dict_total = get_frequency_words(data_sentence_string_set_new, "total")

    regroup_frequency_dict_participant = get_frequency_words(data_sentence_string_set_new, "participant")

    delete!(regroup_frequency_dict_participant, Set([]))

    delete!(regroup_frequency_dict_total, Set([]))

    isfile("$(current_data_path)/regroup_frequency_dict_participant_dh_le.txt") && rm("$(current_data_path)/regroup_frequency_dict_participant_dh_le.txt")
    isfile("$(current_data_path)/regroup_frequency_dict_total_dh_le.txt") && rm("$(current_data_path)/regroup_frequency_dict_total_dh_le.txt")
    log_dictionary(regroup_frequency_dict_participant, "value", "$(current_data_path)/regroup_frequency_dict_participant_dh_le.txt")
    log_dictionary(regroup_frequency_dict_total, "value", "$(current_data_path)/regroup_frequency_dict_total_dh_le.txt")

    # word_dict = Dict()

    # word_names = ["A$(i)" for i = 1 :length(keys(regroup_frequency_dict_participant))]

    # words_meaning = collect(keys(regroup_frequency_dict_participant))

    # for i = 1:length(word_names)
    #     word_dict[words_meaning[i]] = word_names[i]
    # end

    id_length_dict = get_sentence_length_dict(data_sentence_string_set_new)

    isfile("$(current_data_path)/id_length_dh_le.txt") && rm("$(current_data_path)/id_length_dh_le.txt")
    log_dictionary(id_length_dict, "value", "$(current_data_path)/id_length.txt")

    # shuffled_included_stressors = args.included_stressors


    # for key in collect(keys(data_sentence_string_set_new))
    #     for visit in data_sentence_string_set_new[key]

    #         shuffled_included_stressors = shuffle(shuffled_included_stressors)

    #         order_map = Dict(shuffled_included_stressors[i] => i for i in 1:length(shuffled_included_stressors))

    #         # order_map = Dict(included_stressors[i] => i for i in 1:length(included_stressors))
    #         for group in visit
    #             # sort!(group, by = x->parse(Int, x[4:end]))
    #             sort!(group, by = x -> order_map[x])
                
    #         end
    #         # sort!(visit, by = x->parse(Int, x[1][4:end]))
    #         sort!(visit, by = x -> order_map[x[1]])
            
    #     end
        
    # end

    word_to_vec_dict = word_to_vector(events_list, collect(keys(regroup_frequency_dict_participant)), true, true)


    # log word to vec dict 
    isfile("$(current_data_path)/word_to_vec_dict_dh_le.txt") && rm("$(current_data_path)/word_to_vec_dict_dh_le.txt")
    log_dictionary(word_to_vec_dict, "key", "$(current_data_path)/word_to_vec_dict_dh_le.txt")

    vector_sequence_dict = convert_to_vector_sequence(data_sentence_string_set_new, word_to_vec_dict, true)


    #save the list of keys in a txt file
    isfile("$(current_data_path)/vector_sequence_dict_dh_le.txt") && rm("$(current_data_path)/vector_sequence_dict_dh_le.txt")
    log_dictionary(vector_sequence_dict, "value", "$(current_data_path)/vector_sequence_dict_dh_le.txt")

    #save the arguments of the function in a textfile
    # values_to_prepare_sequences_dict = Dict("i_s" => included_events[4:end], "e_f" => args.end_flag, "g_oh_e_f" => args.group_one_hot_encoding_flag, "m_s_s" => args.min_streak_size, "g_t" => args.grouping_threshold, "d_s" => date_str, "m_d" => args.min_days_dh_to_report_pos)
    
    # log_dictionary(values_to_prepare_sequences_dict, "key", "$(current_data_path)/values_to_prepare_sequences_dict_dh_le.txt")
    

    CSV.write("$(current_data_path)/data.csv",df)

    
    return current_data_path, df, word_to_vec_dict, vector_sequence_dict, data_sentence_string_set_new

end



function calculate_chi_square(df)

    cross_tab_matrix = create_cross_tab_matrix(df)

    expected_values_matrix = fill(0.0, 2, 2)

    expected_values_matrix[1,1] = (sum(cross_tab_matrix[1,:]) * sum(cross_tab_matrix[:, 1]))/sum(cross_tab_matrix)
    expected_values_matrix[1,2] = (sum(cross_tab_matrix[1,:]) * sum(cross_tab_matrix[:, 2]))/sum(cross_tab_matrix)
    expected_values_matrix[2,1] = (sum(cross_tab_matrix[2,:]) * sum(cross_tab_matrix[:, 1]))/sum(cross_tab_matrix)
    expected_values_matrix[2,2] = (sum(cross_tab_matrix[2,:]) * sum(cross_tab_matrix[:, 2]))/sum(cross_tab_matrix)

    return sum(((cross_tab_matrix .- expected_values_matrix).^2) ./ expected_values_matrix)
end

function create_cross_tab_matrix(df)
    cross_tab_matrix = fill(0, 2, 2)

    cross_tab_matrix[1,1] = sum((df[!, 1].==0) .& (df[!, 2].==0))
    cross_tab_matrix[1,2] = sum((df[!, 1].==0) .& (df[!, 2].==1))
    cross_tab_matrix[2,1] = sum((df[!, 1].==1) .& (df[!, 2].==0))
    cross_tab_matrix[2,2] = sum((df[!, 1].==1) .& (df[!, 2].==1))

    return cross_tab_matrix
end




function chi_square_statistic(df)
    stressors_list = names(df)[4:end]

    p = length(stressors_list)
    chi_square_test_matrix = fill(0.0, p, p)


    for i = 1:p
        for j = i:p
            if i == j 
                selected_df = DataFrame(hcat(df[:, stressors_list[i]], df[:, stressors_list[i]]), [stressors_list[i], string(stressors_list[i], "_")])
            else
                selected_df = select(df, [Symbol(stressors_list[i]), Symbol(stressors_list[j])])
            end
            chi_square_test_matrix[i, j] = isnan(calculate_chi_square(selected_df)) ? 0 : calculate_chi_square(selected_df)
        end
    end

    chi_square_test_matrix = chi_square_test_matrix .+ chi_square_test_matrix'

    pvalue_matrix = ccdf.(Chisq.(1), chi_square_test_matrix) #.* LinearAlgebra.UpperTriangular(ones(size(chi_square_test_matrix)))
    # .+ chi_square_test_matrix' 
    return chi_square_test_matrix, pvalue_matrix


end

function get_chi_square_metric(df, method = "count")

    id_chi_square_matrix_dict, id_pvalues_matrix_dict = chi_square_statistic_per_participant(df)


    if method == "count"

        chi_square_metric_matrix = count_significant_chi_square_statistics(id_chi_square_matrix_dict)

        chi_square_metric_matrix = chi_square_metric_matrix .* .!(Bool.(LinearAlgebra.Diagonal(ones(size(chi_square_metric_matrix)))))


    elseif method == "median"

        chi_square_metric_matrix = median_chi_square_statistics(id_chi_square_matrix_dict)

        chi_square_metric_matrix = chi_square_metric_matrix .* .!(Bool.(LinearAlgebra.Diagonal(ones(size(chi_square_metric_matrix)))))

    end

    return chi_square_metric_matrix
end



function chi_square_statistic_per_participant(df)

    id_chi_square_matrix_dict = Dict() 
    id_pvalues_matrix_dict = Dict()

    ids = collect(Set(df.id))

    for id in ids
        id_chi_square_matrix_dict[id], id_pvalues_matrix_dict[id] = chi_square_statistic(df[df.id .== id, :])
    end


    id_chi_square_matrix_dict, id_pvalues_matrix_dict
end



function count_significant_chi_square_statistics(id_chi_square_matrix_dict)

    ids = collect(keys(id_chi_square_matrix_dict))

    count_significant_chi_square_matrix = fill(0.0, size(id_chi_square_matrix_dict[ids[1]],1), size(id_chi_square_matrix_dict[ids[1]],2))

    for id in ids
        count_significant_chi_square_matrix = count_significant_chi_square_matrix .+ Int.(id_chi_square_matrix_dict[id] .>  3.841)
    end

    return count_significant_chi_square_matrix
end



function make_dict_of_df(df)

    id_df_dict = Dict()
    for i = 1:size(df, 1)
        if haskey(id_df_dict, df.id[i])
            id_df_dict[df.id[i]] = vcat(id_df_dict[df.id[i]], DataFrame(df[i, 3:end]))
        else
            id_df_dict[df.id[i]] = DataFrame(df[i, 3:end])
        end
    end

    return id_df_dict
end



function create_word_sequence_stressors(df, stressors_list, data_dict)
    data_sequence_indexes_dict = Dict()

    for key in keys(data_dict)
        data_sequence_indexes_dict[key] = []

        for row = 1:size(data_dict[key], 1)
            visit = []
            for col in  stressors_list
                
                if data_dict[key][row, col] == 1
                    push!(visit, col)
                end
            end
            # if length(visit) != 0
            #     push!(visit, "end")
            # end
            
            push!(data_sequence_indexes_dict[key], Set(visit))
        end
    end

    return data_sequence_indexes_dict
end



function remove_empty_sequences(data_sentence_string_set)

    for (key, value) in data_sentence_string_set

        if length(data_sentence_string_set[key]) == 0
            delete!(data_sentence_string_set, participant)
        end
            
    end

    return data_sentence_string_set
end 




function log_dictionary(dict, sorting_rule, filepath)

    """
    log_dictionary(dict::Dict, sorting_rule::String, filepath::String)

    Write a dictionary's key-value pairs to a file, sorted according to a specified rule.

    # Arguments
    - `dict::Dict`: A dictionary with key-value pairs that you want to log.
    - `sorting_rule::String`: A string that specifies the sorting rule. It can be either `"value"`, 
    which will sort the dictionary by values in descending order, or `"key and value"`, 
    which will sort the dictionary first by values in descending order, and then by keys in ascending order.
    - `filepath::String`: The path to the file where you want to log the dictionary.

    # Examples
    dict = Dict("a" => 1, "b" => 2, "c" => 3)
    log_dictionary(dict, "value", "log.txt")
    """

    dict_array = []


    if sorting_rule =="value"
        dict_array = sort(collect(dict), by = x->-x[2])
    elseif sorting_rule =="key and value"
        dict_array = sort(collect(dict), by = x->(-x[2], x[1]))
    else
        dict_array = dict
    end

    open(filepath, "a") do file

        for (key, value) in dict_array 
            println(file, "$(key) => $(value)")
        end
    end

end




function regroup_based_on_chi_square(data_sentence_string_set, count_significant_chi_square_matrix, threshold, included_stressors = nothing)
    
    data_sentence_string_set_new = Dict()
    
    for key in keys(data_sentence_string_set)
        data_sentence_string_set_new[key] = []
        for group in data_sentence_string_set[key]
            # @show group
            if length(collect(group))>= 2
                dependency_matrix = get_pairwise_chi_square_matrix(collect(group), count_significant_chi_square_matrix, included_stressors)
                # @show dependency_matrix

                # @show get_meaningful_subgroup_indexes(dependency_matrix, threshold)
                push!(data_sentence_string_set_new[key], get_meaningful_subgroup_indexes(dependency_matrix, threshold, collect(group)))
                # @show group
                # @show data_sentence_string_set_new[key]
            else
                push!(data_sentence_string_set_new[key], [collect(group)])
            end
        end
    end

    return data_sentence_string_set_new
end




function get_pairwise_chi_square_matrix(group, count_significant_chi_square_matrix, included_stressors = nothing)

    group_length = length(group)

    group_specific_chi_square_matrix = fill(0.0, group_length, group_length)

    for i = 1:length(group)
        for j = 1:length(group)
            # index_i = extract_number(group[i])
            # index_j = extract_number(group[j])

            if !isnothing(included_stressors)
                index_i = findfirst(x->x==group[i], included_stressors)
                index_j = findfirst(x->x==group[j], included_stressors)
            end

            

            if index_i != -1 && index_j != -1
                group_specific_chi_square_matrix[i, j] = count_significant_chi_square_matrix[index_i, index_j]

            end
        end
    end


    y_labels = group
    x_labels = group

    # heatmap(x_labels, y_labels, reverse(group_specific_chi_square_matrix, dims = 1), xticks = :all, yticks = :all, size =(1000, 700), xrotation = 90)
    group_specific_chi_square_matrix
end 



function get_meaningful_subgroup_indexes(dependency_matrix, threshold, group)

    already_grouped_indexes = []
    subgroups = []

    indexes = collect(1:size(dependency_matrix, 1))
    
    max_chi_statistics = maximum(dependency_matrix)


    subgroup = []

    while max_chi_statistics >= threshold
        
        # @show dependency_matrix
        new_indexes = findall(x->x== maximum(dependency_matrix), dependency_matrix)[1]
        # @show new_indexes
        if length(subgroup) == 0
            if !(new_indexes[1] in already_grouped_indexes)
                push!(already_grouped_indexes, new_indexes[1])
                push!(subgroup, new_indexes[1])
            end
            if !(new_indexes[2] in already_grouped_indexes)
                push!(already_grouped_indexes, new_indexes[2])
                push!(subgroup, new_indexes[2])
            end
        else
            divide_flag = false
            for i in Tuple(new_indexes)
                for j in subgroup
                    if dependency_matrix[i,j] < threshold
                        divide_flag = true
                        push!(subgroups, Set(subgroup))
                        subgroup = []
                        break
                    end
                end

                if divide_flag
                    break
                end

                
            end

            if !divide_flag
                if !(new_indexes[1] in already_grouped_indexes)
                    push!(already_grouped_indexes, new_indexes[1])
                    push!(subgroup, new_indexes[1])
                end
                if !(new_indexes[2] in already_grouped_indexes)
                    push!(already_grouped_indexes, new_indexes[2])
                    push!(subgroup, new_indexes[2])
                end
            end
            
        end

        dependency_matrix[new_indexes[1], new_indexes[2]] = -Inf
        dependency_matrix[new_indexes[2], new_indexes[1]] = -Inf

        max_chi_statistics = maximum(dependency_matrix)
    end

    not_included_indexes = collect(setdiff(Set(indexes), Set(already_grouped_indexes)))
    [push!(subgroups, Set(index)) for index in not_included_indexes]

    # @show subgroups
    return convert_indexes_to_names(subgroups, group)
end

function convert_indexes_to_names(subgroups, group)
    if !isscalar(subgroups)
        return [convert_indexes_to_names(x, group) for x in subgroups]
    else
        return group[subgroups]
    end
end


function isscalar(x)
    return !(typeof(x) <: AbstractArray) && !(typeof(x) <: Set) 
end


function get_frequency_words(data_dict, method = "participant")

    group_frequency_dict = Dict()

    for id in keys(data_dict)
        for position in data_dict[id]
            for group in position
                if haskey(group_frequency_dict, Set(group))
                    group_frequency_dict[Set(group)] += 1
                    # if method == "participant"
                    #     exit = true
                    #     break
                    # end
                else
                    group_frequency_dict[Set(group)] = 1
                    # if method == "participant"
                    #     exit = true
                    #     break
                    # end
                end


            end
            
            # if exit
            #     break
            # end
        end
    end
    

    if method == "participant"

        for key in keys(group_frequency_dict)
            group_frequency_dict[key] = 0
            for id in keys(data_dict)
                exit = false
                for position in data_dict[id]
                    for group in position
                        if Set(group) == key
                            group_frequency_dict[key] += 1
                            if method == "participant"
                                exit = true
                                break
                            end
                        end
                    end
                    if exit
                        break
                    end
                end
            end
        end

        
    end
    return group_frequency_dict

end




function get_sentence_length_dict(data_dict)

    id_length_dict = Dict()

    for id in keys(data_dict)
        id_length_dict[id] = 0
        for position in data_dict[id]
            for word in position
                id_length_dict[id] += 1
            end
        end
    end

    println("Median of sequence length: ", Statistics.median(collect(values(id_length_dict))))
    println("Mean of sequence length: ", mean(collect(values(id_length_dict))))

    return id_length_dict
end




function word_to_vector(included_stressors, words, end_flag=false, group_one_hot_encoding_flag=true)


    # included_stressors may not be all found in words!
    # included_stressors = included_stressors[findall(x -> Set([x]) in words, included_stressors)]

    if group_one_hot_encoding_flag
        if end_flag
            vector_length = length(words) + 1
        else
            vector_length = length(words)
        end
    else
        if Set(["dh_many"]) in words
            if end_flag
                vector_length = length(included_stressors) + 2
            else
                vector_length = length(included_stressors) + 1
            end
        else
            if end_flag
                vector_length = length(included_stressors) + 1
            else
                vector_length = length(included_stressors)
            end
        end
        # vector_length = count(col -> startswith(col, "dh") || startswith(col, "le"), names(df)) + 1 #plus one for _many
    end

    word_to_vec_dict = Dict()

    combination_idx = length(included_stressors) + 1 #starting point for grouped items
    
    # combination_idx = 80 #starting point for grouped items

    cnt =1
    for word in words


        word_array_format = collect(word)

        if word_array_format == ["dh_many"]
            vec = fill(0.0, vector_length)
            vec[combination_idx] = 1
            combination_idx += 1
            word_to_vec_dict[Set(word_array_format)] = vec
            continue
        end

        vec = fill(0.0, vector_length)

        if !group_one_hot_encoding_flag

            dh_number_list = get_index_of_grouped_words(word_array_format, included_stressors)
            vec[dh_number_list] .= 1

        else
            if length(word_array_format) > 1
                vec[combination_idx] = 1
                combination_idx += 1

            else
                dh_number_list = get_index_of_grouped_words(word_array_format, included_stressors)
                
                if !isnothing(dh_number_list[1])
                    vec[dh_number_list] .= 1
                end
            end
        end

        word_to_vec_dict[Set(word_array_format)] = vec
        cnt += 1
    end

    if end_flag
        vec = fill(0.0, vector_length)
        vec[combination_idx] = 1
        word_to_vec_dict[Set(["end"])] = vec
    end
    return word_to_vec_dict
end



function get_index_of_grouped_words(word_array_format, included_stressors)
    item_number_list = []
    for i in  1: length(word_array_format)
        
        item_number = findfirst(x->x==word_array_format[i], included_stressors)

        push!(item_number_list, item_number)

    end
    item_number_list
end



function convert_to_vector_sequence(word_sequence_dict, word_to_vec_dict, end_flag=false)
    vector_sequence_dict = Dict()

  
        for participant in collect(keys(word_sequence_dict))
            vector_sequence_dict[participant] = []
            for group in word_sequence_dict[participant]
                for word in group
                    if word != []
                        push!(vector_sequence_dict[participant], word_to_vec_dict[Set(word)])
                    end

                end
                # add end token and assume this is the one with only last one equal to one
                if end_flag
                    end_vec = copy(collect(values(word_to_vec_dict))[1])
                    end_vec .= 0
                    end_vec[end] = 1
                    push!(vector_sequence_dict[participant], end_vec)
                end

            end

        end


    return vector_sequence_dict
end


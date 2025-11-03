
@testset "Vocabulary" begin
   

    dataset = NanoDataset(DATAFILE, 1)

    # These should be in the vocabulary
    required_in_vocab = ['n', 'f', 'w', 'E', 'Z', 'o', '\'', 'B', 'C', ':', '\n', 'h', 'i',  'r','t',  'q', 'M', 'K', 
        'J', 'P', 'I', ';', 'H', 'a', 'c', 'p', 'W', '$', '-', 'T', '.', 'U',
        'Y', 'v', 'x', 'u', 'V', 'b', '!', '&', 'd', 'e', 'X', '?', 'D', 
        'A', 'j', 's', 'y', 'k', ',', 'R', 'G', ' ', 'F', 'N', 'O', 'Q', 'm', 
        'S', 'z', 'L', '3', 'g', 'l']
  
    # Test that all required characters are in the dataset
    for c in required_in_vocab
        @test c ∈ keys(dataset.ch_to_int)
    end

    @test get_vocab_size(dataset) == length(required_in_vocab)
end

@testset "DataLoader" begin

    block_size = 16
    d = NanoDataset(DATAFILE, 16)

    # Test that the getobs function fetches sequences shifted by 1
    xb, yb = getobs(d, 1)
    @test xb == d.data[1:1+block_size-1]
    @test yb == d.data[2:2+block_size-1]


    dl = DataLoader(d)
    
    # Test that when iterating over the DataLoader we still get shifted sequences
    # This is basically the test above
    for (ix, item) in enumerate(dl)
        @test item[1] == (d.data[ix:ix+d.block_size-1], d.data[ix+1:ix+d.block_size])
        ix > 10 && break
    end

end


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
end

@testset "DataLoader" begin

    d = NanoDataset(DATAFILE, 16)
    @test getobs(d, 2) == d.data[2:2+16-1]

end

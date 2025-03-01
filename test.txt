$headers = @{
    "Authorization" = "Bearer YOUR_TOKEN_HERE"
    "Content-Type" = "application/json"
}

# Define 5 test inputs (questions)
$testInputs = @(
    "What are the key features of Jina embeddings for text processing?",
    "Describe a futuristic space mission to Mars in detail.",
    "How can AI improve customer service in 2025?",
    "What is the history of space exploration up to 2025?",
    "Explain the concept of quantum computing in simple terms."
)

$body = @{
    model = "jina_ai/jina-embeddings-v2-base-en"
    input = $testInputs
} | ConvertTo-Json -Depth 10

$response = Invoke-WebRequest `
    -Uri "http://your-litellm-endpoint" `
    -Method Post `
    -Headers $headers `
    -Body $body `
    -ContentType "application/json"

# Parse the JSON response
$jsonResponse = $response.Content | ConvertFrom-Json

# Check the structure of the response and extract embeddings dynamically
if ($jsonResponse.data -and $jsonResponse.data.embedding) {
    # If embeddings are in a list of dictionaries (each with an "embedding" field)
    $embeddings = $jsonResponse.data.embedding
    $batchSize = $embeddings.Count
    if ($embeddings.Count -gt 0) {
        $embeddingDim = $embeddings[0].Count
    } else {
        Write-Output "No embeddings found in the response."
        exit 1
    }
    
    Write-Output "Output shape: $batchSize,$embeddingDim"
    for ($i = 0; $i -lt $batchSize; $i++) {
        Write-Output "First 5 values of embedding $i: $($embeddings[$i][0..4])"
    }
    Write-Output "Embeddings differ check (first vs. last): $($embeddings[0] -ne $embeddings[$batchSize - 1])"
} elseif ($jsonResponse.data -and $jsonResponse.data.Count -gt 0) {
    # If embeddings are a flat list, assume they are concatenated (e.g., batchSize * embeddingDim)
    $totalValues = $jsonResponse.data.Count
    if ($totalValues % 768 -eq 0) {  # Assuming 768 is the embedding dimension for Jina
        $batchSize = $totalValues / 768
        $embeddingDim = 768
        Write-Output "Output shape: $batchSize,$embeddingDim"
        Write-Output "First 5 values of data: $($jsonResponse.data[0..4])"
        Write-Output "Last 5 values of data: $($jsonResponse.data[-5..-1])"
        Write-Output "Embeddings differ check (first vs. last block): $(Compare-Object -ReferenceObject $jsonResponse.data[0..767] -DifferenceObject $jsonResponse.data[768..($totalValues-1)])"
    } else {
        Write-Output "Unexpected number of values in data: $totalValues. Cannot determine shape."
    }
} else {
    Write-Output "Unexpected response structure: $jsonResponse"
}

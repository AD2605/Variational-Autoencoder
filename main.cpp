#include <iostream>
#include <torch/torch.h>
#include "Autoencoder.h"

int main() {
    torch::manual_seed(1234);
    auto device_type = torch::kCUDA;
    torch::Device device(device_type);

    int64_t h = 400;
    int64_t z = 20;
    int64_t image = 28*28;
    int64_t batch = 50;

    VAE model(image, h, z);

    auto dataset = torch::data::datasets::MNIST("path to dataset").
            map(torch::data::transforms::Normalize<>(0.1307, 0.3081));
    size_t dataset_size = dataset.size().value();
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::StreamSampler>
            (std::move(dataset), batch);
    std::setprecision(16);
    model.train(true);
    model.train_model(100, device, train_loader, model, dataset_size, image);
    return 0;
}

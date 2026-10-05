from .convlstm import ConvLSTM, ConvLSTMDecoder, ConvLSTMEncoder
from .forecaster import ConvLSTMForecaster, DualEncoderForecaster
from .pixel_lstm import PixelLSTM


def input_channels(config: dict) -> int:
    return len(config['input_bands']) + int(config['lagged_rzsm'])


def build_model(config: dict):
    if config['decoupled_history']:
        return DualEncoderForecaster(config['hidden_dims'], config['kernel_sizes'],
                                     config['forecast_horizon'], config['dropout'])
    if config['model'] == 'convlstm':
        return ConvLSTMForecaster(input_channels(config), config['hidden_dims'],
                                  config['kernel_sizes'], config['forecast_horizon'],
                                  config['dropout'], doy_channels=2 * config['day_of_year'])
    if config['model'] == 'pixel_lstm':
        return PixelLSTM(input_channels(config), config['hidden_dim'], config['num_layers'],
                         config['forecast_horizon'], config['dropout'])
    raise ValueError(f"unknown model: {config['model']}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

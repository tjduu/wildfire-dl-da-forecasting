
# import pytest
# from unittest.mock import patch, MagicMock

# from src.generative.training import train_vae

# @pytest.fixture
# def setup_model():
#     # Assume nn.Module, optim.Optimizer are correctly set up here
#     model = MagicMock()
#     optimizer = MagicMock()
#     scheduler = MagicMock()
#     train_loader = MagicMock()
#     val_loader = MagicMock()
#     return model, optimizer, scheduler, train_loader, val_loader

# def test_train_vae_calls_train_and_validate(setup_model):
#     model, optimizer, scheduler, train_loader, val_loader = setup_model
#     n_epochs = 5
#     device = "cpu"
    
#     with patch('src.generative.training.train') as mock_train, \
#          patch('src.generative.training.validate') as mock_validate:
#         train_vae(
#             n_epochs=n_epochs,
#             model=model,
#             optimizer=optimizer,
#             scheduler=None,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             model_save_path=None,
#             use_liveloss=False,
#             device=device
#         )

#         # Check train and validate are called n_epochs times
#         assert mock_train.call_count == n_epochs
#         assert mock_validate.call_count == n_epochs
        
#         mock_train.assert_called_with(
#             model=model,
#             optimizer=optimizer,
#             data_loader=train_loader,
#             tepoch=MagicMock(),
#             curr_epoch=1,
#             n_epochs=n_epochs,
#             device=device,
#             logs=True
#         )
#         mock_validate.assert_called_with(
#             model=model,
#             data_loader=val_loader,
#             device=device
#         )
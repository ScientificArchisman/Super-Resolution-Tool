from model import Generator, Discriminator
from loss import GeneratoradversarialLoss, DiscriminatorLoss, ContentLoss
import torch 
torch.set_num_threads(1)
import config
import numpy as np
from imageloader import create_dataloaders
from logger import setup_logging

device = config.DEVICE

# Define the model name and configuration
model_name = "SRGAN"
model_config = {
    "Generator Input Channels": config.GENERATOR_INPUT_CHANNELS,
    "Generator Output Channels": config.GENERATOR_OUTPUT_CHANNELS,
    "Num Upsample Blocks": config.NUM_UPSAMPLE_BLOCKS,
    "Num Residual Blocks": config.NUM_RESIDUAL_BLOCKS,
    "Discriminator Input Channels": config.DISCRIMINATOR_INPUT_CHANNELS,
    "Learning Rate": 1e-4,
    "Batch Size": config.BATCH_SIZE,
    "Number of Epochs": config.N_EPOCHS
}


# Create the models
discriminator = Discriminator(input_channels=config.DISCRIMINATOR_INPUT_CHANNELS).to(device)
generator = Generator(in_channels=config.GENERATOR_INPUT_CHANNELS, 
                      out_channels=config.GENERATOR_OUTPUT_CHANNELS, 
                      num_upsample_blocks=config.NUM_UPSAMPLE_BLOCKS, 
                      num_residual_blocks=config.NUM_RESIDUAL_BLOCKS).to(device)

# Create the optimizers
generator_optim = torch.optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# Create the loss functions
generator_loss = GeneratoradversarialLoss().to(device)
discriminator_loss = DiscriminatorLoss().to(device)
Content_loss = ContentLoss().to(device)

# Create the dataloaders
train_loader, validation_loader, test_loader = create_dataloaders(low_res_dir=config.LOW_RES_FOLDER, 
                                                                high_res_dir=config.HIGH_RES_FOLDER, 
                                                                batch_size=config.BATCH_SIZE, 
                                                                num_workers=config.NUM_WORKERS)

# Setup logging
batch_logger, epoch_logger = setup_logging(log_dir='logs', model_name=model_name, config=model_config)

def train(n_epochs=config.N_EPOCHS, generator=generator, discriminator=discriminator,
          generator_optim=generator_optim, discriminator_optim=discriminator_optim,
          generator_loss=generator_loss, discriminator_loss=discriminator_loss,
          content_loss=Content_loss, train_loader=train_loader, validation_loader=validation_loader):
    
    generator_train_losses, discriminator_train_losses, content_train_losses = [], [], []
    generator_val_losses, discriminator_val_losses = [], []
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        train_loss_G = 0.0
        train_loss_D = 0.0
        train_content_loss = 0.0
        train_adversarial_loss = 0.0
        
        # Training phase
        for idx, (lr, hr) in enumerate(train_loader):
            print(f"Epoch {epoch+1}/{n_epochs}, Batch {idx+1}/{len(train_loader)}")
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Train Discriminator
            discriminator_optim.zero_grad()
            real_outputs = discriminator(hr)
            fake_images = generator(lr)
            fake_outputs = discriminator(fake_images.detach())
            loss_D = discriminator_loss(real_outputs, fake_outputs)
            loss_D.backward()
            discriminator_optim.step()

            # Train Generator
            generator_optim.zero_grad()
            fake_outputs = discriminator(fake_images)
            adversarial_loss = generator_loss(fake_outputs)
            content_loss_value = content_loss(fake_images, hr)
            loss_G = content_loss_value + 0.006 * adversarial_loss
            loss_G.backward()
            generator_optim.step()

            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()
            train_content_loss += content_loss_value.item()
            train_adversarial_loss += adversarial_loss.item()
            
            # Log batch metrics
            batch_logger.info(f"{epoch+1}, {idx+1}, {loss_G.item():.4f}, {loss_D.item():.4f}, {content_loss_value.item():.4f}, {adversarial_loss.item():.4f}")

        avg_train_loss_G = train_loss_G / len(train_loader)
        avg_train_loss_D = train_loss_D / len(train_loader)
        avg_train_content_loss = train_content_loss / len(train_loader)
        avg_train_adversarial_loss = train_adversarial_loss / len(train_loader)

        generator_train_losses.append(avg_train_loss_G)
        discriminator_train_losses.append(avg_train_loss_D)
        content_train_losses.append(avg_train_content_loss)

        # Validation phase
        generator.eval()
        discriminator.eval()
        val_loss_G = 0.0
        val_loss_D = 0.0

        with torch.no_grad():
            for lr, hr in validation_loader:
                lr = lr.to(device)
                hr = hr.to(device)
                
                # Validation loss for Discriminator
                real_outputs = discriminator(hr)
                fake_images = generator(lr)
                fake_outputs = discriminator(fake_images)
                loss_D = discriminator_loss(real_outputs, fake_outputs)
                val_loss_D += loss_D.item()

                # Validation loss for Generator
                loss_G_adv = generator_loss(fake_outputs)
                loss_G_vgg = content_loss(fake_images, hr)
                loss_G = loss_G_vgg + 0.006 * loss_G_adv
                val_loss_G += loss_G.item()

        avg_val_loss_G = val_loss_G / len(validation_loader)
        avg_val_loss_D = val_loss_D / len(validation_loader)

        generator_val_losses.append(avg_val_loss_G)
        discriminator_val_losses.append(avg_val_loss_D)

        print(f"[Epoch {epoch+1}/{n_epochs}] "
              f"Train Loss G: {avg_train_loss_G:.4f}, D: {avg_train_loss_D:.4f}, Content Loss: {avg_train_content_loss:.4f}, Adversarial Loss: {avg_train_adversarial_loss:.4f} | "
              f"Val Loss G: {avg_val_loss_G:.4f}, D: {avg_val_loss_D:.4f}")
        
        # Log epoch metrics
        epoch_logger.info(f"{epoch+1}, {avg_train_loss_G:.4f}, {avg_train_loss_D:.4f}, {avg_train_content_loss:.4f}, {avg_train_adversarial_loss:.4f}, {avg_val_loss_G:.4f}, {avg_val_loss_D:.4f}")

        # Save the model
        torch.save(generator.state_dict(), config.GENERATOR_PATH)
        torch.save(discriminator.state_dict(), config.DISCRIMINATOR_PATH)

        np.save('losses/SRGAN/generator_train_losses.npy', generator_train_losses)
        np.save('losses/SRGAN/discriminator_train_losses.npy', discriminator_train_losses)
        np.save('losses/SRGAN/content_train_losses.npy', content_train_losses)
        np.save('losses/SRGAN/generator_val_losses.npy', generator_val_losses)
        np.save('losses/SRGAN/discriminator_val_losses.npy', discriminator_val_losses)

if __name__ == "__main__":
    train()
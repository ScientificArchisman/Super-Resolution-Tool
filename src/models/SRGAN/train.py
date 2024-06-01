from model import Generator, Discriminator
from loss import VGGLoss, GeneratoradversarialLoss, DiscriminatorLoss
import torch 
torch.set_num_threads(1)
import config
from imageloader import create_dataloaders


device = config.DEVICE

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
generator_adversarial_loss = GeneratoradversarialLoss().to(device)
discriminator_loss = DiscriminatorLoss().to(device)
vgg = VGGLoss(device=device).to(device)

# Create the dataloaders
train_loader, validation_loader, test_loader = create_dataloaders(low_res_dir=config.LOW_RES_FOLDER, 
                                                                high_res_dir=config.HIGH_RES_FOLDER, 
                                                                batch_size=config.BATCH_SIZE, 
                                                                num_workers=config.NUM_WORKERS)

def train(n_epochs = config.N_EPOCHS):
    
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        train_loss_G = 0.0
        train_loss_D = 0.0
        
        # Training phase
        for lr, hr in train_loader:
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
            adversarial_loss = generator_adversarial_loss(fake_outputs)
            content_loss = vgg(fake_images, hr)
            loss_G = content_loss + 0.006 * adversarial_loss
            loss_G.backward()
            generator_optim.step()

            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()

        
        avg_train_loss_G = train_loss_G / len(train_loader)
        avg_train_loss_D = train_loss_D / len(train_loader)


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
                loss_G_adv = generator_adversarial_loss(fake_outputs)
                loss_G_vgg = vgg(fake_images, hr)
                loss_G = loss_G_vgg + 0.006 * loss_G_adv
                val_loss_G += loss_G.item()

        avg_val_loss_G = val_loss_G / len(validation_loader)
        avg_val_loss_D = val_loss_D / len(validation_loader)

        print(f"[Epoch {epoch+1}/{n_epochs}] "
            f"Train Loss G: {avg_train_loss_G:.4f}, D: {avg_train_loss_D:.4f} | "
            f"Val Loss G: {avg_val_loss_G:.4f}, D: {avg_val_loss_D:.4f}")
        
        # Save the model
        torch.save(generator.state_dict(), config.GENERATOR_PATH)
        torch.save(discriminator.state_dict(), config.DISCRIMINATOR_PATH)


        



if __name__ == "__main__":
    train()

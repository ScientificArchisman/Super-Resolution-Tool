from model import Generator, Discriminator
from loss import VGGLoss, GeneratoradversarialLoss, DiscriminatorLoss
import torch 
import config


device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator_optim = torch.optim.Adam(Generator.parameters(), lr=1e-4)
discriminator_optim = torch.optim.Adam(Discriminator.parameters(), lr=1e-4)

discriminator = Discriminator(input_channels=config.DISCRIMINATOR_INPUT_CHANNELS)
generator = Generator(in_channels=config.GENERATOR_INPUT_CHANNELS, 
                      out_channels=config.GENERATOR_OUTPUT_CHANNELS, 
                      num_upsample_blocks=config.NUM_UPSAMPLE_BLOCKS, 
                      num_residual_blocks=config.NUM_RESIDUAL_BLOCKS).to(device)


generator_adversarial_loss = GeneratoradversarialLoss().to(device)
discriminator_loss = DiscriminatorLoss().to(device)
vgg = VGGLoss().to(device)

for epoch in range(config.N_EPOCHS):
    for idx, (lr, hr) in enumerate(dataloader):
        lr = lr.to(device)
        hr = hr.to(device)

        # Train the discriminator
        discriminator_optim.zero_grad()
        real_outputs = discriminator(hr)
        fake_outputs = discriminator(generator(lr).detach())

        d_loss = discriminator_loss(real_outputs, fake_outputs) 
        d_loss.backward()
        discriminator_optim.step()

        # Train the generator   
        generator_optim.zero_grad()
        fake_images = generator(lr)
        fake_outputs = discriminator(fake_images)
        gen_loss = generator_adversarial_loss(fake_outputs)
        perceptual_loss = vgg(fake_images, hr)

        gen_loss = gen_loss + 0.006 * perceptual_loss
        gen_loss.backward()
        generator_optim.step()


        print(f"[Epoch {epoch}/{config.N_EPOCHS}] 
              [Batch {idx}/{len(dataloader)}] [Discrim loss: {d_loss.item()}] [Gen loss: {gen_loss.item()}]")

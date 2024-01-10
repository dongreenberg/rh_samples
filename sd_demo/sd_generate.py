import runhouse as rh
from diffusers import StableDiffusionPipeline


def sd_generate(prompt):
    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to("cuda")
    return model(prompt).images


if __name__ == "__main__":
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", open_ports=[443], server_connection_type="tls", den_auth=True)
    remote_sd_generate = rh.function(sd_generate).to(gpu, env=["./", "torch", "transformers", "diffusers"])
    imgs = remote_sd_generate("A hot dog made out of matcha.")
    imgs[0].show()
    remote_sd_generate.share("den_tester")

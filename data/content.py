
class ContentSources:
    """
    List of content sources with titles and URLs.
    """

    def __init__(self):
      self.file_links = [
      {
          "title": "Attention Is All You Need",
          "url": "https://arxiv.org/pdf/1706.03762"
      },
      {
          "title": "Adding Conditional Control to Text-to-Image Diffusion Models",
          "url": "https://arxiv.org/pdf/2302.05543"
      },
      {
          "title": "BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding",
          "url": "https://arxiv.org/pdf/1810.04805"
      },
      {
          "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
          "url": "https://arxiv.org/pdf/2201.11903"
      },
      {
          "title": "Denoising Diffusion Probabilistic Models",
          "url": "https://arxiv.org/pdf/2006.11239"
      },
      {
          "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
          "url": "https://arxiv.org/pdf/2112.10752"
      },
      {
          "title": "Image-to-Image Translation with Conditional Adversarial Networks",
          "url": "https://arxiv.org/pdf/1611.07004"
      },
      {
          "title": "Instruction Tuning for Large Language Models- A Survey",
          "url": "https://arxiv.org/pdf/2308.10792"
      },
      {
          "title": "Llama 2- Open Foundation and Fine-Tuned Chat Models",
          "url": "https://arxiv.org/pdf/2307.09288"
      },
      {
          "title": "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
          "url": "https://arxiv.org/pdf/1703.10593"
      },
      {
          "title": "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS",
          "url": "https://arxiv.org/pdf/1511.06434"
      },
      {
          "title": "OFFLINE REINFORCEMENT LEARNING WITH IMPLICIT Q-LEARNING",
          "url": "https://arxiv.org/pdf/2110.06169"
      },
      {
         "title": "PAPER2CODE: AUTOMATING CODE GENERATION FROM SCIENTIFIC PAPERS IN MACHINE LEARNING",
         "url": "https://arxiv.org/pdf/2504.17192"
      },
      {
         "title": "Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning",
         "url": "https://www.arxiv.org/pdf/2512.07461"
      },
      {
         "title": "Conservative Q-Learning for Offline Reinforcement Learning",
         "url": "https://arxiv.org/abs/2006.04779"
      }
    ]
    def get_pdf_urls(self):
       return [item['url'] for item in self.file_links]

if __name__ == "__main__":
   print(ContentSources().get_pdf_urls())

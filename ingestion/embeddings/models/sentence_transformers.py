from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import loguru

from ingestion.dataprep.chunkers.base import Chunk

class SentenceTransformerEmbedder:
    def __init__(self,model_name:str='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model:SentenceTransformer = None
        self.dimension = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            test_embedding = self.model.encode('test')
            self.dimension = len(test_embedding)
        except Exception as e:
            raise

    def encode(self,chunks:List[Chunk],batch_size:int = 32) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        if not chunks:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        # Determine input type
        if isinstance(chunks[0], Chunk):
            texts = [c.content for c in chunks]
        elif isinstance(chunks[0], str):
            texts = chunks
        else:
            raise ValueError("encode() expects a list of Chunk or str")

        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise
    
    def encode_single(self,text:str) -> np.ndarray:
        """Encode single text string"""
        return self.encode(text)[0]
    
    def get_dimension(self) -> int:
        """Get dimension"""
        return self.dimension


if __name__ == '__main__':
    embedder = SentenceTransformerEmbedder()

    chunks = [Chunk(id='59a4437b-4297-421a-9ccf-5f844863582b', content='[DO NOT PUBLISH] \n \nIN THE UNITED STATES COURT OF APPEALS \n \nFOR THE ELEVENTH CIRCUIT \n________________________ \n \nNo. 18-15233  \nNon-Argument Calendar \n________________________ \n \nD.C. Docket No. 3:17-cr-00221-MMH-JBT-5 \n \nUNITED STATES OF AMERICA,  \n \n                                                                                 \nPlaintiff-Appellee, \n \n                                                             versus \n \nXAVIER THOMAS ALEXANDER,  \n \n                                                                                 \nDefendant-Appellant. \n________________________ \n \nAppeal from the United States District Court \nfor the Middle District of Florida \n________________________ \n(February 6, 2020) \nBefore JORDAN, GRANT, and TJOFLAT, Circuit Judges. \nPER CURIAM: \nCase: 18-15233     Date Filed: 02/06/2020     Page: 1 of 4', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 1}), Chunk(id='f744c441-a701-44c3-ab98-7acdb023f6c0', content='2 \n \n \nXavier Alexander appeals his 120-month sentence for conspiracy to \ndistribute cocaine, challenging the district court’s determination that he is a career \noffender for sentencing purposes based on his two prior state felony convictions \nfor sale of cocaine and possession of cocaine with intent to sell.  See Fla. Stat. \n§ 893.13.  On appeal, Alexander argues that these crimes cannot be “controlled \nsubstance offenses” that trigger the career-offender designation under the \nSentencing Guidelines because (1) the more serious offense of Florida cocaine \ntrafficking is not considered a controlled substance offense, and (2) the Florida \nstatute defining his offenses, § 893.13(1) of the Florida Statutes, does not contain a \nmens rea requirement as to the illicit nature of the substance involved.  These \narguments are foreclosed by the plain language of the Sentencing Guidelines and \nby binding precedent. \n \nWe review de novo the question whether a defendant qualifies as a career \noffender under the Sentencing Guidelines.  United States v. Pridgeon, 853 F.3d \n1192, 1198 n.1 (11th Cir. 2017).  To be a career offender, a defendant must have \ntwo or more prior felony convictions that qualify as “either a crime of violence or a \ncontrolled substance offense.”  United States Sentencing Commission, Guidelines \nManual § 4B1.1(a).  The Guidelines define a “controlled substance offense” as a \nfelony that involves “the manufacture, import, export, distribution, or dispensing of', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 2}), Chunk(id='0787b6ee-c72e-4bd1-ae11-814aab4f988d', content='felony that involves “the manufacture, import, export, distribution, or dispensing of \na controlled substance (or a counterfeit substance) or the possession of a controlled \nCase: 18-15233     Date Filed: 02/06/2020     Page: 2 of 4', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 2}), Chunk(id='0b7fdca9-118b-4ab7-84b3-6de28762c7e2', content='3 \n \nsubstance (or a counterfeit substance) with intent to manufacture, import, export, \ndistribute, or dispense.”  Id. § 4B1.2(b). \n \nIn interpreting these provisions, we apply the usual rules of statutory \nconstruction, beginning with the plain language of the guideline.  United States v. \nShannon, 631 F.3d 1187, 1189 (11th Cir. 2011).  In Shannon, therefore, we held \nthat a conviction for Florida cocaine trafficking involving only the purchase of \ncocaine was not a “controlled substance offense” under § 4B1.2(b) because the \npurchase of cocaine “does not necessarily give rise to actual or constructive \npossession” of the drug under Florida law, and the act of purchasing cocaine is not \ncovered by the plain language of the guideline.  Id. at 1188–90.  We noted that a \nviolation of the same Florida drug trafficking statute that involved possession with \nintent to distribute cocaine—rather than purchase with intent to distribute—would \nmeet the definition of a controlled substance offense.  Id. at 1190 & n.3.  Contrary \nto Alexander’s argument, whether a prior state felony is a controlled substance \noffense for purposes of the career-offender guideline depends on whether the state \noffense meets the definition of that term in § 4B1.2(b)—not on the seriousness of \nthe offense or the severity of the penalty under state law.  Cf. id. at 1190–91 \n(Marcus, J., specially concurring).  \nIn United States v. Smith, we determined that a violation of § 893.13(1) of', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 3}), Chunk(id='79bfc7df-7576-4975-9af6-95b51e47b773', content='(Marcus, J., specially concurring).  \nIn United States v. Smith, we determined that a violation of § 893.13(1) of \nthe Florida Statutes—which provides that, with exceptions not relevant here, “a \nCase: 18-15233     Date Filed: 02/06/2020     Page: 3 of 4', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 3}), Chunk(id='fa96c8f2-bde5-4be3-89f2-5181ffb57e13', content='4 \n \nperson may not sell, manufacture, or deliver, or possess with intent to sell, \nmanufacture, or deliver, a controlled substance”—squarely meets the definition of \na “controlled substance offense” under the Guidelines.  775 F.3d 1262, 1267 (11th \nCir. 2014).  We specifically rejected the argument that because the Florida statute \ndoes not require proof that the defendant knew that the substance was illegal, a \nviolation of § 893.13(1) should not qualify as a controlled substance offense.  Id.; \nsee also Pridgeon, 853 F.3d at 1197–98.  As we explained in Smith, no “element of \nmens rea with respect to the illicit nature of the controlled substance is expressed \nor implied by” the Guidelines definition of “controlled substance offense.”  Smith, \n775 F.3d at 1267.  We are bound by this precedent.  See, e.g., United States v. \nHarris, 941 F.3d 1048, 1057 (11th Cir. 2019).   \n \nThe district court appropriately applied the career-offender enhancement \nwhen calculating Alexander’s Guidelines sentencing range because his Florida \nfelony convictions for sale of cocaine and possession of cocaine with intent to sell \nqualify as controlled substance offenses under the Guidelines.  We therefore affirm \nAlexander’s conviction and sentence. \nAFFIRMED. \nCase: 18-15233     Date Filed: 02/06/2020     Page: 4 of 4', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 4})]

    embeds = embedder.encode(chunks)
    print(embeds)
from dataclasses import dataclass
from pathlib import PosixPath
from typing import List, Any, Dict, Optional
import uuid
from ingestion.dataprep.chunkers.base import Chunk, Chunker

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ingestion.dataprep.annotators.pipeline import PageAnnotator

class LangChainChunker(Chunker):
    def __init__(self,chunk_size:int=1500,chunk_overlap:int=150,separators:Optional[List[str]]=None,length_function=len):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
            length_function=length_function,
        )
    
    def split(self, pages:List[PageAnnotator]) -> List[Chunk]:
        chunks:List[Chunk] = []
        for page in pages:
            text = page.normalized_text
        
            # Build comprehensive metadata from PageAnnotator attributes
            base_md = {}
            
            # Add source as string
            if hasattr(page, 'source') and page.source:
                base_md['source'] = str(page.source)
            
            # Always include page number
            base_md['page_number'] = page.page_number
            
            # Add caption metadata (contains court, date, parties, case_name)
            if page.caption:
                base_md.update(page.caption)
            
            # Add other structured fields if they exist and are not None/empty
            if page.procedural_posture:
                base_md['procedural_posture'] = page.procedural_posture
                
            if page.statutes_guidelines:
                base_md['statutes_guidelines'] = page.statutes_guidelines
                
            if page.precedents:
                base_md['precedents'] = page.precedents
                
            if page.holdings:
                base_md['holdings'] = page.holdings
                
            if page.disposition:
                base_md['disposition'] = page.disposition
                
            if page.calendar_status:
                base_md['calendar_status'] = page.calendar_status
                
            if page.extra:
                base_md['extra'] = page.extra
            
            # Build a Document and split
            doc_obj = Document(page_content=text, metadata=base_md)
            docs = self.splitter.split_documents([doc_obj])  # returns chunked Documents with copied metadata
            
            for d in docs:
                md = dict(d.metadata or {})
                chunks.append(Chunk(id=str(uuid.uuid4()), content=d.page_content, metadata=md))
    
        return chunks


    
if __name__ == '__main__':
    chunker = LangChainChunker()
    doc = [
    PageAnnotator(page_number=1, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption={'court': 'United States Court Of Appeals', 'date': 'February 6, 2020', 'parties': {'plaintiff': 'United States of America', 'defendant': 'Xavier Thomas Alexander'}, 'case_name': 'United States of America v. Xavier Thomas Alexander'}, procedural_posture='Appeal from the United States District Court for the Middle District of Florida', statutes_guidelines=[], precedents=[], holdings=[], disposition=None, normalized_text='In The United States Court Of Appeals\n\nFor The Eleventh Circuit ________________________\n\nNo. 18-15233 Non-Argument Calendar ________________________\n\n\nUnited States Of America,\n\nPlaintiff-Appellee,\n\nversus\n\nXavier Thomas Alexander,\n\nDefendant-Appellant. ________________________\n\nAppeal from the United States District Court for the Middle District of Florida ________________________ (February 6, 2020) Before Jordan, Grant, and Tjoflat, Circuit Judges. Per Curiam:', calendar_status='Non-Argument Calendar', extra={}), 
    PageAnnotator(page_number=2, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption=None, procedural_posture=None, statutes_guidelines=['Fla. Stat. § 893.13', '§ 4B1.1(a)', '§ 893.13', '§ 893.13(1)'], precedents=[], holdings=['On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.', 'To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.'], disposition=None, normalized_text='Xavier Alexander appeals his 120-month sentence for conspiracy to distribute cocaine, challenging the district court’s determination that he is a career offender for sentencing purposes based on his two prior state felony convictions for sale of cocaine and possession of cocaine with intent to sell. See Fla. Stat. § 893.13. On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.13(1) of the Florida Statutes, does not contain a mens rea requirement as to the illicit nature of the substance involved. These arguments are foreclosed by the plain language of the Sentencing Guidelines and by binding precedent.\n\nWe review de novo the question whether a defendant qualifies as a career offender under the Sentencing Guidelines. United States v. Pridgeon, 853 F.3d 1192, 1198 n.1 (11th Cir. 2017). To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.” United States Sentencing Commission, Guidelines Manual § 4B1.1(a). The Guidelines define a “controlled substance offense” as a felony that involves “the manufacture, import, export, distribution, or dispensing of a controlled substance (or a counterfeit substance) or the possession of a controlled', calendar_status=None, extra={}), 
    PageAnnotator(page_number=3, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption=None, procedural_posture=None, statutes_guidelines=['§ 4B1.2(b)', '§ 893.13(1)'], precedents=[], holdings=[], disposition=None, normalized_text='substance (or a counterfeit substance) with intent to manufacture, import, export, distribute, or dispense.” Id. § 4B1.2(b).\n\nIn interpreting these provisions, we apply the usual rules of statutory construction, beginning with the plain language of the guideline. United States v. Shannon, 631 F.3d 1187, 1189 (11th Cir. 2011). In Shannon, therefore, we held that a conviction for Florida cocaine trafficking involving only the purchase of cocaine was not a “controlled substance offense” under § 4B1.2(b) because the purchase of cocaine “does not necessarily give rise to actual or constructive possession” of the drug under Florida law, and the act of purchasing cocaine is not covered by the plain language of the guideline. Id. at 1188–90. We noted that a violation of the same Florida drug trafficking statute that involved possession with intent to distribute cocaine—rather than purchase with intent to distribute—would meet the definition of a controlled substance offense. Id. at 1190 & n.3. Contrary to Alexander’s argument, whether a prior state felony is a controlled substance offense for purposes of the career-offender guideline depends on whether the state offense meets the definition of that term in § 4B1.2(b)—not on the seriousness of the offense or the severity of the penalty under state law. Cf. id. at 1190–91 (Marcus, J., specially concurring). In United States v. Smith, we determined that a violation of § 893.13(1) of the Florida Statutes—which provides that, with exceptions not relevant here, “a', calendar_status=None, extra={}), 
    PageAnnotator(page_number=4, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption=None, procedural_posture=None, statutes_guidelines=['§ 893.13(1)'], precedents=[], holdings=[], disposition='Affirmed', normalized_text='person may not sell, manufacture, or deliver, or possess with intent to sell, manufacture, or deliver, a controlled substance”—squarely meets the definition of a “controlled substance offense” under the Guidelines. 775 F.3d 1262, 1267 (11th Cir. 2014). We specifically rejected the argument that because the Florida statute does not require proof that the defendant knew that the substance was illegal, a violation of § 893.13(1) should not qualify as a controlled substance offense. Id.; see also Pridgeon, 853 F.3d at 1197–98. As we explained in Smith, no “element of mens rea with respect to the illicit nature of the controlled substance is expressed or implied by” the Guidelines definition of “controlled substance offense.” Smith, 775 F.3d at 1267. We are bound by this precedent. See, e.g., United States v. Harris, 941 F.3d 1048, 1057 (11th Cir. 2019).\n\nThe district court appropriately applied the career-offender enhancement when calculating Alexander’s Guidelines sentencing range because his Florida felony convictions for sale of cocaine and possession of cocaine with intent to sell qualify as controlled substance offenses under the Guidelines. We therefore affirm Alexander’s conviction and sentence. Affirmed.', calendar_status=None, extra={})]

    result = chunker.split(doc)
    print(result)


# RESULT

[
    Chunk(id='9c332f3c-df32-4cfb-bec4-cee4b7bae672', content='In The United States Court Of Appeals\n\nFor The Eleventh Circuit ________________________\n\nNo. 18-15233 Non-Argument Calendar ________________________\n\n\nUnited States Of America,\n\nPlaintiff-Appellee,\n\nversus\n\nXavier Thomas Alexander,\n\nDefendant-Appellant. ________________________\n\nAppeal from the United States District Court for the Middle District of Florida ________________________ (February 6, 2020) Before Jordan, Grant, and Tjoflat, Circuit Judges. Per Curiam:', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 1, 'court': 'United States Court Of Appeals', 'date': 'February 6, 2020', 'parties': {'plaintiff': 'United States of America', 'defendant': 'Xavier Thomas Alexander'}, 'case_name': 'United States of America v. Xavier Thomas Alexander', 'procedural_posture': 'Appeal from the United States District Court for the Middle District of Florida', 'calendar_status': 'Non-Argument Calendar'}), 
    Chunk(id='37343e77-52e7-4a50-8e42-d95fb381703c', content='Xavier Alexander appeals his 120-month sentence for conspiracy to distribute cocaine, challenging the district court’s determination that he is a career offender for sentencing purposes based on his two prior state felony convictions for sale of cocaine and possession of cocaine with intent to sell. See Fla. Stat. § 893.13. On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.13(1) of the Florida Statutes, does not contain a mens rea requirement as to the illicit nature of the substance involved. These arguments are foreclosed by the plain language of the Sentencing Guidelines and by binding precedent.', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 2, 'statutes_guidelines': ['Fla. Stat. § 893.13', '§ 4B1.1(a)', '§ 893.13', '§ 893.13(1)'], 'holdings': ['On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.', 'To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.']}), 
    Chunk(id='640ce82f-3439-4e1b-abe7-a9f67937b276', content='We review de novo the question whether a defendant qualifies as a career offender under the Sentencing Guidelines. United States v. Pridgeon, 853 F.3d 1192, 1198 n.1 (11th Cir. 2017). To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.” United States Sentencing Commission, Guidelines Manual § 4B1.1(a). The Guidelines define a “controlled substance offense” as a felony that involves “the manufacture, import, export, distribution, or dispensing of a controlled substance (or a counterfeit substance) or the possession of a controlled', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 2, 'statutes_guidelines': ['Fla. Stat. § 893.13', '§ 4B1.1(a)', '§ 893.13', '§ 893.13(1)'], 'holdings': ['On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.', 'To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.']}), Chunk(id='cedff9a7-a1d5-4d04-a2c7-525e4e55af83', content='substance (or a counterfeit substance) with intent to manufacture, import, export, distribute, or dispense.” Id. § 4B1.2(b).', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 3, 'statutes_guidelines': ['§ 4B1.2(b)', '§ 893.13(1)']}), 
    Chunk(id='64a48f65-80a0-40de-826b-413979aabeb6', content='In interpreting these provisions, we apply the usual rules of statutory construction, beginning with the plain language of the guideline. United States v. Shannon, 631 F.3d 1187, 1189 (11th Cir. 2011). In Shannon, therefore, we held that a conviction for Florida cocaine trafficking involving only the purchase of cocaine was not a “controlled substance offense” under § 4B1.2(b) because the purchase of cocaine “does not necessarily give rise to actual or constructive possession” of the drug under Florida law, and the act of purchasing cocaine is not covered by the plain language of the guideline. Id. at 1188–90. We noted that a violation of the same Florida drug trafficking statute that involved possession with intent to distribute cocaine—rather than purchase with intent to distribute—would meet the definition of a controlled substance offense. Id. at 1190 & n.3. Contrary to Alexander’s argument, whether a prior state felony is a controlled substance offense for purposes of the career-offender guideline depends on whether the state offense meets the definition of that term in § 4B1.2(b)—not on the seriousness of the offense or the severity of the penalty under state law. Cf. id. at 1190–91 (Marcus, J., specially concurring). In United States v. Smith, we determined that a violation of § 893.13(1) of the Florida Statutes—which provides that, with exceptions not relevant here, “a', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 3, 'statutes_guidelines': ['§ 4B1.2(b)', '§ 893.13(1)']}), 
    Chunk(id='fc9963bb-7b1a-4467-85d1-f4a33ae685a8', content='person may not sell, manufacture, or deliver, or possess with intent to sell, manufacture, or deliver, a controlled substance”—squarely meets the definition of a “controlled substance offense” under the Guidelines. 775 F.3d 1262, 1267 (11th Cir. 2014). We specifically rejected the argument that because the Florida statute does not require proof that the defendant knew that the substance was illegal, a violation of § 893.13(1) should not qualify as a controlled substance offense. Id.; see also Pridgeon, 853 F.3d at 1197–98. As we explained in Smith, no “element of mens rea with respect to the illicit nature of the controlled substance is expressed or implied by” the Guidelines definition of “controlled substance offense.” Smith, 775 F.3d at 1267. We are bound by this precedent. See, e.g., United States v. Harris, 941 F.3d 1048, 1057 (11th Cir. 2019).\n\nThe district court appropriately applied the career-offender enhancement when calculating Alexander’s Guidelines sentencing range because his Florida felony convictions for sale of cocaine and possession of cocaine with intent to sell qualify as controlled substance offenses under the Guidelines. We therefore affirm Alexander’s conviction and sentence. Affirmed.', metadata={'source': '/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', 'page_number': 4, 'statutes_guidelines': ['§ 893.13(1)'], 'disposition': 'Affirmed'})]
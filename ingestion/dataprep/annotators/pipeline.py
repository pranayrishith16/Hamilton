from __future__ import annotations
import calendar
from curses import panel
from dataclasses import dataclass, field
from os import name
from pathlib import PosixPath
from typing import List, Optional, Dict, Any, Tuple
import re

from numpy import number
from tomlkit import date
from transformers import DeepseekVLForConditionalGeneration

from ingestion.dataprep.parsers.interfaces import RawPage

@dataclass
class PageAnnotator:
    page_number:int = None
    source:Optional[str] = None
    caption: Optional[Dict[str, Any]] = None
    procedural_posture: Optional[str] = None
    statutes_guidelines: List[str] = None
    precedents: List[Dict[str, Any]] = None
    holdings: List[str] = None
    disposition: Optional[str] = None
    normalized_text: str = None
    calendar_status: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

class PageLegalAnnotator:
    def drop_caption_noise(self, text: str) -> str:
        # Remove “DO NOT PUBLISH”, page numbers, and repeated docket lines
        text = re.sub(r'(?im)^\s*\[DO NOT PUBLISH\]\s*$', '', text)
        text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
        text = re.sub(r'(?m)^D\.C\.\s*Docket\s*No\.\s*[^\n]+\n?', '', text, flags=re.I)
        return text.strip()

    def dehyphenate(self, text: str) -> str:
        return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    def normalize_headings(self, text: str) -> str:
        def to_title(m):
            s = m.group(0)
            return s.title() if len(s.split()) <= 8 else s
        return re.sub(r'\b([A-Z][A-Z \-]{2,})\b', to_title, text)

    def extract_caption(self, text: str) -> Optional[Dict[str, Any]]:
        block: Dict[str, Any] = {}
        m = re.search(r'\bIN THE (.+?COURT OF APPEALS.*?)\r?\n', text, flags=re.I)
        if m:
            block['court'] = m.group(1).strip()
        m = re.search(r'\bD\.C\.\s*Docket\s*No\.\s*([A-Za-z0-9\-\.:]+)', text, flags=re.I)
        if m:
            block['docket'] = m.group(1).strip()
        m = re.search(
            r'\((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\)',
            text
        )
        if m:
            block['date'] = m.group(0).strip('()')

        parties: Dict[str, str] = {}
        m = re.search(r'\bUNITED STATES OF AMERICA\b', text, flags=re.I)
        if m:
            parties['plaintiff'] = 'United States of America'
        m = re.search(r'\b([A-Z][A-Z \.\'-]+?),\s*\r?\n\s*Defendant', text, flags=re.I)
        if m:
            parties['defendant'] = m.group(1).title()
        if parties:
            block['parties'] = parties
            if 'plaintiff' in parties and 'defendant' in parties:
                block['case_name'] = f"{parties['plaintiff']} v. {parties['defendant']}"

        return block or None

    def extract_procedural(self, text: str) -> Optional[str]:
        m = re.search(r'\b(Appeal from [^\n\r]+?)(?=\s*_{5,}|\s*\(|\s*Before)', text, flags=re.I)
        return m.group(1).strip() if m else None

    def extract_statutes(self, text: str) -> List[str]:
        raw = set(re.findall(r'(?:USSG\s*)?§+\s*[\dA-Za-z\.\-\(\)]+', text))
        raw |= set(re.findall(r'\b[A-Z][a-z]+\.\s*Stat\.\s*§\s*[\dA-Za-z\.\-\(\)]+', text))
        cleaned = {s.rstrip('.').replace('  ', ' ').strip() for s in raw}
        return sorted(cleaned)

    def extract_precedents(self, text: str) -> List[Dict[str, Any]]:
        cites: List[Dict[str, Any]] = []
        patt = (
            r'([A-Z][A-Za-z\.\'& ]+ v\. [A-Z][A-Za-z\.\'& ]+),\s*'
            r'(\d{1,4} [A-Za-z\.]+ \d+)(?:\s*\(\d{4}\))?'
            r'(?:,\s*(\d+))?'
            r'\s*\(([^\)]+)\)'
        )
        for m in re.finditer(patt, text):
            cites.append({
                'case': m.group(1).strip(),
                'reporter': m.group(2).strip(),
                'pin': m.group(3).strip() if m.group(3) else None,
                'court_year': m.group(4).strip()
            })
        return cites

    def extract_holdings_disposition(self, text: str) -> Tuple[List[str], Optional[str], Optional[str]]:
        holdings: List[str] = []
        patterns = [
            r'We (?:hold|conclude|determine|find|rule)\s+that\s+([^\.]+\.)',
            r'It is (?:well )?(?:established|settled) that [^\.]+\.',
            r'The (?:rule|law|precedent) (?:is|requires|mandates) [^\.]+\.',
            r'(?:Under|Pursuant to) [^,]+?,? [^\.]* must [^\.]+\.',
            r'[A-Z][^\.]* (?:must|cannot) [^\.]+\.',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.I):
                sent = m.group(0).strip()
                if len(sent) > 10 and sent.endswith('.'):
                    holdings.append(sent)

        calendar = None
        if re.search(r'\bNon-Argument Calendar\b', text, flags=re.I):
            calendar = 'Non-Argument Calendar'

        disposition = None
        for pat, label in [
            (r'\bAFFIRMED\b\.?', 'Affirmed'),
            (r'\bREVERSED\s+AND\s+REMANDED\b\.?', 'Reversed and Remanded'),
            (r'\bAFFIRMED\s+IN\s+PART\b\.?', 'Affirmed in Part'),
            (r'\bREVERSED\s+IN\s+PART\b\.?', 'Reversed in Part'),
            (r'\bREVERSED\b\.?', 'Reversed'),
            (r'\bVACATED\b\.?', 'Vacated'),
            (r'\bREMANDED\b\.?', 'Remanded'),
            (r'\bDISMISSED\b\.?', 'Dismissed'),
            (r'\bGRANTED\b\.?', 'Granted'),
            (r'\bDENIED\b\.?', 'Denied'),
        ]:
            if re.search(pat, text, flags=re.I):
                disposition = label
                break

        return holdings, disposition, calendar

    def annotate_page(self, page: RawPage) -> PageAnnotator:
        txt = page.text or ''
        txt = self.drop_caption_noise(txt)
        txt = self.dehyphenate(txt)
        txt = self.normalize_headings(txt)

        caption = self.extract_caption(txt)
        posture = self.extract_procedural(txt)
        statutes = self.extract_statutes(txt)
        precedents = self.extract_precedents(txt)
        holdings, disposition, calendar = self.extract_holdings_disposition(txt)

        src = ''
        if hasattr(page, 'metadata') and isinstance(page.metadata, dict):
            src = str(page.metadata.get('source', ''))

        return PageAnnotator(
            page_number=page.page_number,
            source=src,
            caption=caption,
            procedural_posture=posture,
            statutes_guidelines=statutes,
            precedents=precedents,
            holdings=holdings,
            disposition=disposition,
            calendar_status=calendar,
            normalized_text=txt
        )
    
    def annotate(self,pages:List[RawPage]) -> List[PageAnnotator]:
        return [self.annotate_page(page) for page in pages]


if __name__ == '__main__':

    input = [RawPage(page_number=1, text='[DO NOT PUBLISH]\n\nIN THE UNITED STATES COURT OF APPEALS\n\nFOR THE ELEVENTH CIRCUIT ________________________\n\nNo. 18-15233 Non-Argument Calendar ________________________\n\nD.C. Docket No. 3:17-cr-00221-MMH-JBT-5\n\nUNITED STATES OF AMERICA,\n\nPlaintiff-Appellee,\n\nversus\n\nXAVIER THOMAS ALEXANDER,\n\nDefendant-Appellant. ________________________\n\nAppeal from the United States District Court for the Middle District of Florida ________________________ (February 6, 2020) Before JORDAN, GRANT, and TJOFLAT, Circuit Judges. PER CURIAM:', metadata={'source': PosixPath('/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf')}), RawPage(page_number=2, text='2\n\nXavier Alexander appeals his 120-month sentence for conspiracy to distribute cocaine, challenging the district court’s determination that he is a career offender for sentencing purposes based on his two prior state felony convictions for sale of cocaine and possession of cocaine with intent to sell. See Fla. Stat. § 893.13. On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.13(1) of the Florida Statutes, does not contain a mens rea requirement as to the illicit nature of the substance involved. These arguments are foreclosed by the plain language of the Sentencing Guidelines and by binding precedent.\n\nWe review de novo the question whether a defendant qualifies as a career offender under the Sentencing Guidelines. United States v. Pridgeon, 853 F.3d 1192, 1198 n.1 (11th Cir. 2017). To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.” United States Sentencing Commission, Guidelines Manual § 4B1.1(a). The Guidelines define a “controlled substance offense” as a felony that involves “the manufacture, import, export, distribution, or dispensing of a controlled substance (or a counterfeit substance) or the possession of a controlled', metadata={'source': PosixPath('/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf')}), RawPage(page_number=3, text='3\n\nsubstance (or a counterfeit substance) with intent to manufacture, import, export, distribute, or dispense.” Id. § 4B1.2(b).\n\nIn interpreting these provisions, we apply the usual rules of statutory construction, beginning with the plain language of the guideline. United States v. Shannon, 631 F.3d 1187, 1189 (11th Cir. 2011). In Shannon, therefore, we held that a conviction for Florida cocaine trafficking involving only the purchase of cocaine was not a “controlled substance offense” under § 4B1.2(b) because the purchase of cocaine “does not necessarily give rise to actual or constructive possession” of the drug under Florida law, and the act of purchasing cocaine is not covered by the plain language of the guideline. Id. at 1188–90. We noted that a violation of the same Florida drug trafficking statute that involved possession with intent to distribute cocaine—rather than purchase with intent to distribute—would meet the definition of a controlled substance offense. Id. at 1190 & n.3. Contrary to Alexander’s argument, whether a prior state felony is a controlled substance offense for purposes of the career-offender guideline depends on whether the state offense meets the definition of that term in § 4B1.2(b)—not on the seriousness of the offense or the severity of the penalty under state law. Cf. id. at 1190–91 (Marcus, J., specially concurring). In United States v. Smith, we determined that a violation of § 893.13(1) of the Florida Statutes—which provides that, with exceptions not relevant here, “a', metadata={'source': PosixPath('/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf')}), RawPage(page_number=4, text='4\n\nperson may not sell, manufacture, or deliver, or possess with intent to sell, manufacture, or deliver, a controlled substance”—squarely meets the definition of a “controlled substance offense” under the Guidelines. 775 F.3d 1262, 1267 (11th Cir. 2014). We specifically rejected the argument that because the Florida statute does not require proof that the defendant knew that the substance was illegal, a violation of § 893.13(1) should not qualify as a controlled substance offense. Id.; see also Pridgeon, 853 F.3d at 1197–98. As we explained in Smith, no “element of mens rea with respect to the illicit nature of the controlled substance is expressed or implied by” the Guidelines definition of “controlled substance offense.” Smith, 775 F.3d at 1267. We are bound by this precedent. See, e.g., United States v. Harris, 941 F.3d 1048, 1057 (11th Cir. 2019).\n\nThe district court appropriately applied the career-offender enhancement when calculating Alexander’s Guidelines sentencing range because his Florida felony convictions for sale of cocaine and possession of cocaine with intent to sell qualify as controlled substance offenses under the Guidelines. We therefore affirm Alexander’s conviction and sentence. AFFIRMED.', metadata={'source': PosixPath('/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf')})]

    annotator = PageLegalAnnotator()
    print(annotator.annotate(input))


# output




[
    PageAnnotator(page_number=1, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption={'court': 'United States Court Of Appeals', 'date': 'February 6, 2020', 'parties': {'plaintiff': 'United States of America', 'defendant': 'Xavier Thomas Alexander'}, 'case_name': 'United States of America v. Xavier Thomas Alexander'}, procedural_posture='Appeal from the United States District Court for the Middle District of Florida', statutes_guidelines=[], precedents=[], holdings=[], disposition=None, normalized_text='In The United States Court Of Appeals\n\nFor The Eleventh Circuit ________________________\n\nNo. 18-15233 Non-Argument Calendar ________________________\n\n\nUnited States Of America,\n\nPlaintiff-Appellee,\n\nversus\n\nXavier Thomas Alexander,\n\nDefendant-Appellant. ________________________\n\nAppeal from the United States District Court for the Middle District of Florida ________________________ (February 6, 2020) Before Jordan, Grant, and Tjoflat, Circuit Judges. Per Curiam:', calendar_status='Non-Argument Calendar', extra={}), 
    PageAnnotator(page_number=2, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption=None, procedural_posture=None, statutes_guidelines=['Fla. Stat. § 893.13', '§ 4B1.1(a)', '§ 893.13', '§ 893.13(1)'], precedents=[], holdings=['On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.', 'To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.'], disposition=None, normalized_text='Xavier Alexander appeals his 120-month sentence for conspiracy to distribute cocaine, challenging the district court’s determination that he is a career offender for sentencing purposes based on his two prior state felony convictions for sale of cocaine and possession of cocaine with intent to sell. See Fla. Stat. § 893.13. On appeal, Alexander argues that these crimes cannot be “controlled substance offenses” that trigger the career-offender designation under the Sentencing Guidelines because (1) the more serious offense of Florida cocaine trafficking is not considered a controlled substance offense, and (2) the Florida statute defining his offenses, § 893.13(1) of the Florida Statutes, does not contain a mens rea requirement as to the illicit nature of the substance involved. These arguments are foreclosed by the plain language of the Sentencing Guidelines and by binding precedent.\n\nWe review de novo the question whether a defendant qualifies as a career offender under the Sentencing Guidelines. United States v. Pridgeon, 853 F.3d 1192, 1198 n.1 (11th Cir. 2017). To be a career offender, a defendant must have two or more prior felony convictions that qualify as “either a crime of violence or a controlled substance offense.” United States Sentencing Commission, Guidelines Manual § 4B1.1(a). The Guidelines define a “controlled substance offense” as a felony that involves “the manufacture, import, export, distribution, or dispensing of a controlled substance (or a counterfeit substance) or the possession of a controlled', calendar_status=None, extra={}), 
    PageAnnotator(page_number=3, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption=None, procedural_posture=None, statutes_guidelines=['§ 4B1.2(b)', '§ 893.13(1)'], precedents=[], holdings=[], disposition=None, normalized_text='substance (or a counterfeit substance) with intent to manufacture, import, export, distribute, or dispense.” Id. § 4B1.2(b).\n\nIn interpreting these provisions, we apply the usual rules of statutory construction, beginning with the plain language of the guideline. United States v. Shannon, 631 F.3d 1187, 1189 (11th Cir. 2011). In Shannon, therefore, we held that a conviction for Florida cocaine trafficking involving only the purchase of cocaine was not a “controlled substance offense” under § 4B1.2(b) because the purchase of cocaine “does not necessarily give rise to actual or constructive possession” of the drug under Florida law, and the act of purchasing cocaine is not covered by the plain language of the guideline. Id. at 1188–90. We noted that a violation of the same Florida drug trafficking statute that involved possession with intent to distribute cocaine—rather than purchase with intent to distribute—would meet the definition of a controlled substance offense. Id. at 1190 & n.3. Contrary to Alexander’s argument, whether a prior state felony is a controlled substance offense for purposes of the career-offender guideline depends on whether the state offense meets the definition of that term in § 4B1.2(b)—not on the seriousness of the offense or the severity of the penalty under state law. Cf. id. at 1190–91 (Marcus, J., specially concurring). In United States v. Smith, we determined that a violation of § 893.13(1) of the Florida Statutes—which provides that, with exceptions not relevant here, “a', calendar_status=None, extra={}), 
    PageAnnotator(page_number=4, source='/Users/pranayrishith/Documents/Hamilton/data/2020-02-06_united_states_v._xavier_thomas_alexander.pdf', caption=None, procedural_posture=None, statutes_guidelines=['§ 893.13(1)'], precedents=[], holdings=[], disposition='Affirmed', normalized_text='person may not sell, manufacture, or deliver, or possess with intent to sell, manufacture, or deliver, a controlled substance”—squarely meets the definition of a “controlled substance offense” under the Guidelines. 775 F.3d 1262, 1267 (11th Cir. 2014). We specifically rejected the argument that because the Florida statute does not require proof that the defendant knew that the substance was illegal, a violation of § 893.13(1) should not qualify as a controlled substance offense. Id.; see also Pridgeon, 853 F.3d at 1197–98. As we explained in Smith, no “element of mens rea with respect to the illicit nature of the controlled substance is expressed or implied by” the Guidelines definition of “controlled substance offense.” Smith, 775 F.3d at 1267. We are bound by this precedent. See, e.g., United States v. Harris, 941 F.3d 1048, 1057 (11th Cir. 2019).\n\nThe district court appropriately applied the career-offender enhancement when calculating Alexander’s Guidelines sentencing range because his Florida felony convictions for sale of cocaine and possession of cocaine with intent to sell qualify as controlled substance offenses under the Guidelines. We therefore affirm Alexander’s conviction and sentence. Affirmed.', calendar_status=None, extra={})]
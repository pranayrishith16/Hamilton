from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional, Dict
from lxml import etree

from ingestion.dataprep.parsers.interfaces import Parser, StatuteSection

class XMLParser(Parser):
    """
    Universal XML parser for USLM statute XML (and any similar hierarchical XML).
    Extracts every <section> as a Document, capturing its full hierarchy in metadata.
    """

    def __init__(self,namespace:Optional[str]=None):
        self.ns = {'uslm':namespace} if namespace else {}

    def parse(self,path:str) -> List[StatuteSection]:
        tree = etree.parse(path)
        root = tree.getroot()
        sections: List[StatuteSection] = []

        for sect_info in self._extract_sections(root,[]):
            # Build a stable ID from the path
            path = sect_info["path"]
            # E.g. [("title","17"),("chapter","5"),("section","101"),("subsection","a")]
            id_parts = [f"{tag}-{num}" for tag, num in path if num]
            section_id = "-".join(id_parts)

            # Extract title, chapter, and section numbers from path
            title_num = next((num for tag, num in path if tag == "title"), "")
            chapter_num = next((num for tag, num in path if tag == "chapter"), None)
            section_num = next((num for tag, num in path if tag == "section"), "")

            sections.append(StatuteSection(
                id=section_id,
                title_num=title_num,
                chapter_num=chapter_num,
                section_num=section_num,
                heading=sect_info["heading"],
                content=sect_info["text"],
                metadata={
                    "path": path,
                    "source_file": path
                }
            ))

        return sections

    def _extract_sections(self,elem:etree._Element,path:List[tuple]):
        """
        Recursively traverse XML. Yield dicts with:
          - path: list of (tag, identifier) tuples
          - heading: section heading
          - text: plain-text content
        """

        path = list(path)
        tag = etree.QName(elem).localname.lower()

        if tag in ('title','chapter','subchapter','part','subpart','section'):
            num = elem.get("num")
            if not num and self.ns:
                # use XPath to find uslm:num
                nums = elem.xpath('./uslm:num/text()',namespaces=self.ns)
                num = nums[0] if nums else ""
            path.append((tag,num))
        
        if tag == 'section':
            # Extract heading
            if self.ns:
                headings = elem.xpath('./uslm:heading/text()', namespaces=self.ns)
                heading = headings[0].strip() if headings else ""
            else:
                # No namespace—match by local-name
                headings = elem.xpath('./*[local-name()="heading"]/text()')
                heading = headings[0].strip() if headings else ""

            # Extract all text content
            if self.ns:
                text_nodes = elem.xpath('.//text()[normalize-space()]', namespaces=self.ns)
            else:
                text_nodes = elem.xpath('.//text()[normalize-space()]')
            text = "\n".join(t.strip() for t in text_nodes)

            yield {"path": path, "heading": heading, "text": text}
        
        for child in elem:
            yield from self._extract_sections(child,path)


if __name__ == "__main__":
    xmlparser = XMLParser()
    path = Path(__file__).parent.parent.parent.parent / "data" / "statutes/usc05.xml"
    print(xmlparser.parse(path))

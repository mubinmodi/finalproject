"""
Stage 4: XBRL Extraction + Validation

Parses XBRL files using Arelle to extract canonical financial facts
(concept, context, period, value, units) for validation and KPI extraction.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from utils import config, get_logger, XBRLFact

logger = get_logger("stage4_xbrl")

# Try to import Arelle
try:
    from arelle import Cntlr, ModelManager
    from arelle.ModelValue import qname
    ARELLE_AVAILABLE = True
except ImportError:
    logger.warning("Arelle not available. XBRL extraction will be limited.")
    ARELLE_AVAILABLE = False


class XBRLStage:
    """Extract and validate XBRL facts."""
    
    def __init__(self):
        """Initialize XBRL parser."""
        if ARELLE_AVAILABLE:
            # Initialize Arelle controller
            self.controller = Cntlr.Cntlr(logFileName="logToPrint")
            self.controller.startLogging(logFileName="logToPrint")
            logger.info("Arelle XBRL parser initialized")
        else:
            self.controller = None
            logger.warning("Arelle not initialized")
    
    def process(self, doc_id: str) -> List[Dict]:
        """
        Extract XBRL facts from a document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            List of XBRL fact dictionaries
        """
        logger.info(f"Extracting XBRL facts for {doc_id}")
        
        xbrl_dir = config.paths.raw_dir / doc_id / "xbrl"
        
        if not xbrl_dir.exists():
            logger.warning(f"No XBRL directory found for {doc_id}")
            # Create empty XBRL facts file for validation
            self._save_facts(doc_id, [])
            return []
        
        # Find instance document (typically ends with .xml and contains data)
        instance_files = list(xbrl_dir.glob("*_htm.xml")) or list(xbrl_dir.glob("*.xml"))
        
        if not instance_files:
            logger.warning(f"No XBRL instance documents found in {xbrl_dir}")
            # Create empty XBRL facts file for validation
            self._save_facts(doc_id, [])
            return []
        
        # Use the first instance document
        instance_file = instance_files[0]
        logger.info(f"Processing XBRL instance: {instance_file.name}")
        
        try:
            if ARELLE_AVAILABLE:
                facts = self._extract_facts_arelle(doc_id, instance_file)
            else:
                facts = self._extract_facts_simple(doc_id, instance_file)
            
            # Save facts
            self._save_facts(doc_id, facts)
            
            logger.info(f"✅ Extracted {len(facts)} XBRL facts")
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting XBRL: {e}")
            raise
    
    def _extract_facts_arelle(
        self,
        doc_id: str,
        instance_file: Path
    ) -> List[Dict]:
        """
        Extract facts using Arelle (full featured).
        
        Args:
            doc_id: Document identifier
            instance_file: Path to XBRL instance document
        
        Returns:
            List of fact dictionaries
        """
        facts = []
        
        try:
            # Load XBRL instance
            model_manager = ModelManager.initialize(self.controller)
            model_xbrl = model_manager.load(str(instance_file))
            
            if not model_xbrl:
                logger.error("Failed to load XBRL model")
                return []
            
            # Iterate through facts
            for fact_idx, fact in enumerate(model_xbrl.facts):
                try:
                    # Extract basic fact information
                    concept = fact.qname.localName if fact.qname else None
                    context_id = fact.contextID if fact.contextID else None
                    value = fact.value
                    
                    # Get context information
                    context = model_xbrl.contexts.get(context_id) if context_id else None
                    
                    period = None
                    dimensions = {}
                    
                    if context:
                        # Extract period
                        if context.instantDatetime:
                            period = context.instantDatetime.isoformat()
                        elif context.startDatetime and context.endDatetime:
                            period = f"{context.startDatetime.isoformat()}/{context.endDatetime.isoformat()}"
                        
                        # Extract dimensions
                        if hasattr(context, 'qnameDims'):
                            for dim_qname, member in context.qnameDims.items():
                                dim_name = dim_qname.localName
                                member_name = member.memberQname.localName if hasattr(member, 'memberQname') else str(member)
                                dimensions[dim_name] = member_name
                    
                    # Get units
                    units = None
                    if fact.unitID:
                        unit = model_xbrl.units.get(fact.unitID)
                        if unit and hasattr(unit, 'measures'):
                            # Get first measure
                            if unit.measures and len(unit.measures) > 0:
                                measure_list = unit.measures[0]
                                if measure_list:
                                    units = measure_list[0].localName if hasattr(measure_list[0], 'localName') else str(measure_list[0])
                    
                    # Get decimals
                    decimals = None
                    if hasattr(fact, 'decimals') and fact.decimals is not None:
                        decimals = int(fact.decimals) if fact.decimals != 'INF' else None
                    
                    # Create fact object
                    fact_obj = XBRLFact(
                        fact_id=f"{doc_id}_fact{fact_idx}",
                        doc_id=doc_id,
                        concept=concept or "unknown",
                        context=context_id or "unknown",
                        period=period or "unknown",
                        value=value,
                        units=units,
                        decimals=decimals,
                        dimensions=dimensions if dimensions else None
                    )
                    
                    facts.append(fact_obj.dict())
                    
                except Exception as e:
                    logger.debug(f"Error processing fact {fact_idx}: {e}")
                    continue
            
            # Close model
            model_xbrl.close()
            
        except Exception as e:
            logger.error(f"Arelle extraction failed: {e}")
            raise
        
        return facts
    
    def _extract_facts_simple(
        self,
        doc_id: str,
        instance_file: Path
    ) -> List[Dict]:
        """
        Extract facts using simple XML parsing (fallback).
        
        Args:
            doc_id: Document identifier
            instance_file: Path to XBRL instance document
        
        Returns:
            List of fact dictionaries
        """
        import xml.etree.ElementTree as ET
        
        facts = []
        
        try:
            tree = ET.parse(instance_file)
            root = tree.getroot()
            
            # Get namespace map
            namespaces = dict([node for _, node in ET.iterparse(
                str(instance_file), events=['start-ns']
            )])
            
            fact_idx = 0
            
            # Find all elements with contextRef (these are facts)
            for elem in root.iter():
                if 'contextRef' in elem.attrib:
                    concept = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    context_id = elem.attrib.get('contextRef')
                    value = elem.text
                    units = elem.attrib.get('unitRef')
                    decimals = elem.attrib.get('decimals')
                    
                    fact_obj = XBRLFact(
                        fact_id=f"{doc_id}_fact{fact_idx}",
                        doc_id=doc_id,
                        concept=concept,
                        context=context_id or "unknown",
                        period="unknown",  # Would need to parse context
                        value=value,
                        units=units,
                        decimals=int(decimals) if decimals and decimals.isdigit() else None,
                        dimensions=None
                    )
                    
                    facts.append(fact_obj.dict())
                    fact_idx += 1
        
        except Exception as e:
            logger.error(f"Simple XML extraction failed: {e}")
            raise
        
        return facts
    
    def _save_facts(self, doc_id: str, facts: List[Dict]):
        """Save XBRL facts to JSONL file."""
        output_dir = config.paths.processed_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "xbrl_facts.jsonl"
        
        with open(output_path, 'w') as f:
            for fact in facts:
                f.write(json.dumps(fact) + '\n')
        
        logger.info(f"Saved {len(facts)} XBRL facts to {output_path}")
    
    def get_fact_by_concept(
        self,
        doc_id: str,
        concept: str
    ) -> List[Dict]:
        """
        Get all facts for a specific concept.
        
        Args:
            doc_id: Document identifier
            concept: Concept name (e.g., 'Assets', 'Revenues')
        
        Returns:
            List of matching facts
        """
        facts_path = config.paths.processed_dir / doc_id / "xbrl_facts.jsonl"
        
        if not facts_path.exists():
            return []
        
        matching_facts = []
        
        with open(facts_path, 'r') as f:
            for line in f:
                fact = json.loads(line.strip())
                if concept.lower() in fact['concept'].lower():
                    matching_facts.append(fact)
        
        return matching_facts
    
    def get_key_metrics(self, doc_id: str) -> Dict[str, any]:
        """
        Extract common key financial metrics.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary of key metrics
        """
        # Common XBRL concepts for key metrics
        concept_map = {
            'revenue': ['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax'],
            'net_income': ['NetIncomeLoss', 'ProfitLoss'],
            'assets': ['Assets'],
            'liabilities': ['Liabilities'],
            'equity': ['StockholdersEquity', 'Equity'],
            'cash': ['Cash', 'CashAndCashEquivalentsAtCarryingValue'],
            'eps': ['EarningsPerShareBasic', 'EarningsPerShareDiluted']
        }
        
        metrics = {}
        
        for metric_name, concepts in concept_map.items():
            for concept in concepts:
                facts = self.get_fact_by_concept(doc_id, concept)
                if facts:
                    # Use the most recent fact
                    metrics[metric_name] = facts[0]['value']
                    break
        
        return metrics


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract XBRL facts")
    parser.add_argument("--doc-id", required=True, help="Document ID")
    
    args = parser.parse_args()
    
    stage = XBRLStage()
    facts = stage.process(args.doc_id)
    
    logger.info(f"✅ XBRL extraction complete:")
    logger.info(f"  - Total facts: {len(facts)}")
    
    # Count unique concepts
    concepts = set(f['concept'] for f in facts)
    logger.info(f"  - Unique concepts: {len(concepts)}")
    
    # Show key metrics
    metrics = stage.get_key_metrics(args.doc_id)
    if metrics:
        logger.info("  - Key metrics found:")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value}")


if __name__ == "__main__":
    main()

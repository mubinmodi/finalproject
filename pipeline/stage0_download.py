"""
Stage 0: Download + Organize SEC Filings

Downloads SEC filings (10-K, 10-Q) with XBRL attachments and organizes them
in a consistent directory structure.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from sec_edgar_downloader import Downloader
from utils import config, get_logger, Manifest

logger = get_logger("stage0_download")


class DownloadStage:
    """Download and organize SEC filings."""
    
    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize downloader.
        
        Args:
            user_agent: SEC EDGAR user agent string (required by SEC)
        """
        self.user_agent = user_agent or config.download.user_agent
        self.downloader = Downloader(
            "Project Green Lattern",
            self.user_agent.split()[-1],  # Extract email
            str(config.paths.data_dir)  # Fixed: removed nested directory
        )
        logger.info(f"Initialized downloader with user agent: {self.user_agent}")
    
    def download(
        self,
        ticker: str,
        form_type: str = "10-K",
        limit: int = 1,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[str]:
        """
        Download SEC filings for a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            form_type: Filing form type ('10-K', '10-Q', etc.)
            limit: Number of filings to download
            after: Download filings after this date (YYYY-MM-DD)
            before: Download filings before this date (YYYY-MM-DD)
        
        Returns:
            List of document IDs for downloaded filings
        """
        logger.info(f"Downloading {limit} {form_type} filings for {ticker}")
        
        try:
            # Download filings
            num_downloaded = self.downloader.get(
                form_type,
                ticker,
                limit=limit,
                after=after,
                before=before,
                download_details=True  # Download XBRL and other attachments
            )
            
            logger.info(f"Downloaded {num_downloaded} filings")
            
            # Organize downloaded filings
            doc_ids = self._organize_filings(ticker, form_type)
            
            logger.info(f"Organized {len(doc_ids)} filings: {doc_ids}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error downloading filings: {e}")
            raise
    
    def _organize_filings(self, ticker: str, form_type: str) -> List[str]:
        """
        Organize downloaded filings into standardized structure.
        
        Args:
            ticker: Stock ticker
            form_type: Filing form type
        
        Returns:
            List of document IDs
        """
        # Path where sec-edgar-downloader saves files
        # Note: sec-edgar-downloader keeps the hyphen in form types
        download_path = (
            config.paths.data_dir / 
            config.download.download_dir / 
            ticker / 
            form_type  # Keep hyphen: "10-K" not "10K"
        )
        
        if not download_path.exists():
            logger.warning(f"Download path does not exist: {download_path}")
            return []
        
        doc_ids = []
        
        for filing_dir in sorted(download_path.iterdir()):
            if not filing_dir.is_dir():
                continue
            
            accession_number = filing_dir.name
            
            # Create document ID
            # Extract filing date from directory (sec-edgar-downloader uses accession format)
            # Format: {ticker}_{form_type}_{accession_number}
            doc_id = f"{ticker}_{form_type}_{accession_number}"
            
            # Create standardized directory structure
            doc_dir = config.paths.raw_dir / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Find and copy filing document
            pdf_path = None
            html_path = None
            xbrl_dir = None
            
            # Look for primary filing document
            primary_doc = filing_dir / "primary-document.html"
            if primary_doc.exists():
                html_path = str(doc_dir / "filing.html")
                shutil.copy2(primary_doc, html_path)
                logger.debug(f"Copied HTML: {html_path}")
            
            # Look for PDF version (some filings may have PDFs)
            pdf_found = False
            for file in filing_dir.glob("*.pdf"):
                pdf_path = str(doc_dir / "filing.pdf")
                shutil.copy2(file, pdf_path)
                logger.debug(f"Copied PDF: {pdf_path}")
                pdf_found = True
                break
            
            # If no PDF but HTML exists, convert HTML to PDF
            if not pdf_found and html_path:
                logger.info(f"No PDF found, converting HTML to PDF...")
                try:
                    pdf_path = str(doc_dir / "filing.pdf")
                    self._convert_html_to_pdf(html_path, pdf_path)
                    logger.info(f"✓ Converted HTML to PDF: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Failed to convert HTML to PDF: {e}")
                    pdf_path = None
            
            # Copy XBRL files
            xbrl_files = list(filing_dir.glob("*.xml")) + list(filing_dir.glob("*.xsd"))
            if xbrl_files:
                xbrl_dir = str(doc_dir / "xbrl")
                Path(xbrl_dir).mkdir(exist_ok=True)
                for xbrl_file in xbrl_files:
                    shutil.copy2(xbrl_file, Path(xbrl_dir) / xbrl_file.name)
                logger.debug(f"Copied {len(xbrl_files)} XBRL files")
            
            # Extract filing metadata
            filing_date = self._extract_filing_date(filing_dir)
            
            # Create manifest
            manifest = Manifest(
                doc_id=doc_id,
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                accession_number=accession_number,
                pdf_path=pdf_path,
                html_path=html_path,
                xbrl_dir=xbrl_dir,
                download_timestamp=datetime.now().isoformat()
            )
            
            # Save manifest
            manifest_path = doc_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest.dict(), f, indent=2)
            
            logger.info(f"Created manifest for {doc_id}")
            doc_ids.append(doc_id)
        
        return doc_ids
    
    def _extract_filing_date(self, filing_dir: Path) -> str:
        """
        Extract filing date from downloaded metadata.
        
        Args:
            filing_dir: Directory containing filing
        
        Returns:
            Filing date in YYYY-MM-DD format
        """
        # Try to extract from full-submission.txt metadata
        submission_file = filing_dir / "full-submission.txt"
        if submission_file.exists():
            with open(submission_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                # Look for FILED AS OF DATE
                for line in content.split('\n'):
                    if 'FILED AS OF DATE:' in line or 'FILING DATE:' in line:
                        # Extract date (format: YYYYMMDD)
                        parts = line.split(':')
                        if len(parts) > 1:
                            date_str = parts[1].strip()[:8]
                            if len(date_str) == 8 and date_str.isdigit():
                                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        
        # Fallback: use directory modification time
        return datetime.fromtimestamp(filing_dir.stat().st_mtime).strftime("%Y-%m-%d")
    
    def _convert_html_to_pdf(self, html_path: str, pdf_path: str):
        """
        Convert HTML filing to PDF using weasyprint.
        
        Args:
            html_path: Path to HTML file
            pdf_path: Output PDF path
        """
        try:
            from weasyprint import HTML, CSS
            
            # Read and clean HTML
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Create PDF with weasyprint
            HTML(string=html_content, base_url=str(Path(html_path).parent)).write_pdf(
                pdf_path,
                stylesheets=[CSS(string='@page { size: Letter; margin: 1in; }')]
            )
            
            logger.info(f"✓ Successfully converted HTML to PDF using weasyprint")
            
        except Exception as e:
            logger.error(f"Failed to convert HTML to PDF: {e}")
            raise Exception(f"HTML to PDF conversion failed: {e}")
    
    def list_downloaded(self) -> List[Dict]:
        """
        List all downloaded filings.
        
        Returns:
            List of manifest dictionaries
        """
        manifests = []
        
        for doc_dir in config.paths.raw_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            
            manifest_path = doc_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifests.append(json.load(f))
        
        return manifests


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--form-type", default="10-K", help="Filing form type")
    parser.add_argument("--limit", type=int, default=1, help="Number of filings")
    parser.add_argument("--after", help="Download filings after date (YYYY-MM-DD)")
    parser.add_argument("--before", help="Download filings before date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    downloader = DownloadStage()
    doc_ids = downloader.download(
        ticker=args.ticker,
        form_type=args.form_type,
        limit=args.limit,
        after=args.after,
        before=args.before
    )
    
    logger.info(f"✅ Downloaded {len(doc_ids)} filings")
    for doc_id in doc_ids:
        logger.info(f"  - {doc_id}")


if __name__ == "__main__":
    main()

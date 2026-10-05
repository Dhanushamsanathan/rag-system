"""
Evaluation Benchmark Dataset
Contains 18 Ground-Truth Question -> Relevant Source Passage Pairs covering Bid1 and Bid2.
"""

from typing import List, Dict, Any

EVALUATION_PAIRS: List[Dict[str, Any]] = [
    {
        "id": "eval_01",
        "bid_id": "Bid1",
        "question": "What is the new submission deadline for Bid1 after Addendum 2?",
        "expected_files": ["Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf"],
        "expected_pages": [1],
        "keywords": ["July 9, 2024", "2:00 PM CST", "extend the due date"]
    },
    {
        "id": "eval_02",
        "bid_id": "Bid1",
        "question": "What are the clarifications in Addendum 1 regarding display monitor USB ports?",
        "expected_files": ["Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf"],
        "expected_pages": [1],
        "keywords": ["USB 3.1", "minimum requirement", "Display monitor"]
    },
    {
        "id": "eval_03",
        "bid_id": "Bid2",
        "question": "What is the proposal due date and time for the Dell Laptop PORFP?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf", "Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html"],
        "expected_pages": [1],
        "keywords": ["06/10/2024", "PROPOSAL DUE", "2:00 PM EDT"]
    },
    {
        "id": "eval_04",
        "bid_id": "Bid2",
        "question": "Which affidavits are required for the Maryland Dell laptop solicitation?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf", "Mercury_Affidavit.pdf", "Contract_Affidavit.pdf"],
        "expected_pages": [1, 2],
        "keywords": ["Mercury Affidavit", "Contract Affidavit"]
    },
    {
        "id": "eval_05",
        "bid_id": "Bid2",
        "question": "What model and quantity of laptops are requested in the Maryland PORFP?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf"],
        "expected_pages": [3],
        "keywords": ["Dell Latitude 5550", "30", "SI# CC7802"]
    },
    {
        "id": "eval_06",
        "bid_id": "Bid2",
        "question": "What dock model and quantity are requested in Bid2?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf"],
        "expected_pages": [3],
        "keywords": ["WD22TB4", "Dell Thunderbolt 4 Dock", "30"]
    },
    {
        "id": "eval_07",
        "bid_id": "Bid2",
        "question": "What processor and memory specifications are specified for the Dell Latitude 5550?",
        "expected_files": ["Dell_Laptop_Specs.pdf"],
        "expected_pages": [1],
        "keywords": ["Intel Core Ultra 5 125U", "16 GB", "DDR5", "5600 MT/s"]
    },
    {
        "id": "eval_08",
        "bid_id": "Bid2",
        "question": "What is the delivery timeline requirement after award for the Dell laptops?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf"],
        "expected_pages": [2],
        "keywords": ["45 days of Award", "Delivery within 45 days"]
    },
    {
        "id": "eval_09",
        "bid_id": "Bid2",
        "question": "Who is the agency point of contact for the Maryland State Treasurer PORFP?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf"],
        "expected_pages": [3],
        "keywords": ["Tamaira Hawkins", "thawkins@treasurer.state.md.us", "410-260-7533"]
    },
    {
        "id": "eval_10",
        "bid_id": "Bid2",
        "question": "Where and within how many days should invoices be submitted for the Dell laptop PORFP?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf"],
        "expected_pages": [2],
        "keywords": ["STOaccountspayable@treasurer.state.md.us", "within 10 days of delivering"]
    },
    {
        "id": "eval_11",
        "bid_id": "Bid1",
        "question": "What is the contract term duration and renewal options for Dallas ISD computing devices?",
        "expected_files": ["JA-207652 Student and Staff Computing Devices FINAL.pdf"],
        "expected_pages": [2],
        "keywords": ["three (3) year agreement", "two (2) successive one (1) year"]
    },
    {
        "id": "eval_12",
        "bid_id": "Bid1",
        "question": "What purchasing cooperative is Dallas ISD a member of in this RFP?",
        "expected_files": ["JA-207652 Student and Staff Computing Devices FINAL.pdf"],
        "expected_pages": [19, 20],
        "keywords": ["Educational Purchasing Cooperative of North Texas", "EPCNT"]
    },
    {
        "id": "eval_13",
        "bid_id": "Bid1",
        "question": "What white glove installation and staging services are required for Dallas ISD?",
        "expected_files": ["JA-207652 Student and Staff Computing Devices FINAL.pdf"],
        "expected_pages": [7],
        "keywords": ["software installation", "asset tagging", "etching", "device deployment"]
    },
    {
        "id": "eval_14",
        "bid_id": "Bid1",
        "question": "What is the warranty repair turnaround time SLA expected by Dallas ISD?",
        "expected_files": ["JA-207652 Student and Staff Computing Devices FINAL.pdf"],
        "expected_pages": [4],
        "keywords": ["five (5) business days", "at no cost", "warranty service"]
    },
    {
        "id": "eval_15",
        "bid_id": "Bid1",
        "question": "What are the minimum processor and display specifications for student tablets in Dallas ISD?",
        "expected_files": ["JA-207652 Student and Staff Computing Devices FINAL.pdf"],
        "expected_pages": [5],
        "keywords": ["Processor A13", "10.2\" Retina", "Student Tablet"]
    },
    {
        "id": "eval_16",
        "bid_id": "Bid1",
        "question": "When was the pre-proposal conference scheduled for Dallas ISD RFP JA-207652?",
        "expected_files": ["JA-207652 Student and Staff Computing Devices FINAL.pdf", "Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html"],
        "expected_pages": [2, 1],
        "keywords": ["10-JUN-2024", "06/10/2024", "Pre-Proposal Meeting"]
    },
    {
        "id": "eval_17",
        "bid_id": "Bid2",
        "question": "What is the eMMA project number and master contract number for the Dell laptop solicitation?",
        "expected_files": ["PORFP_-_Dell_Laptop_Final.pdf"],
        "expected_pages": [1],
        "keywords": ["BPM044557", "060B5400007", "Desktop, Laptop and Tablet 2015"]
    },
    {
        "id": "eval_18",
        "bid_id": "Bid1",
        "question": "What form is required for Certificate of Interested Parties in Dallas ISD RFP?",
        "expected_files": ["Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf", "JA-207652 Student and Staff Computing Devices FINAL.pdf"],
        "expected_pages": [4, 17],
        "keywords": ["Form 1295", "Certificate of Interested Parties"]
    }
]

# Kenya questions: cleaning report

Made by `bluff clean-sheets` on 2026-11-19 23:30. Every change is listed row by row in `cleaned/changes.csv` and in the **Changes** tab of `cleaned/cleaned_questions.xlsx`; the cleaned questions are in that workbook's student tabs. Nothing is imported until you run `bluff import-sheet`.

## Sources

- **Student A** (James Gosha): `../Bluff Detector - Kenya Question Sheet student b.xlsx`, tab "Student B"
- **Student B** (Student B): `../Geography_Questions.xlsx`, tab "Sheet1"
- **Student C** (Michael): `../Bluff Detector - Kenya Question Sheet 1.xlsx`, tab "Student C"
- **Student D** (Student D): `../project 2.xlsx`, tab "Student D"

## What was done, per student

| | Student A (James Gosha) | Student B (Student B) | Student C (Michael) | Student D (Student D) |
|---|---|---|---|---|
| AI review, dropped | 3 | 13 | 34 | 31 |
| AI review, edited | 8 | 19 | 7 | 87 |
| AI-written UK twin | 16 | 8 | 5 | 4 |
| AI-written fake | 0 | 0 | 0 | 10 |
| Author filled from student_files.yaml | 0 | 80 | 0 | 51 |
| Excel value shown as typed | 9 | 2 | 1 | 1 |
| added '?' | 0 | 0 | 3 | 14 |
| added 'In Kenya, ' | 16 | 11 | 6 | 1 |
| domain mapped to the brief's six | 46 | 71 | 52 | 51 |
| kept: fake | 20 | 20 | 20 | 10 |
| kept: real | 71 | 60 | 32 | 41 |
| removed: a twin written in the Kenya row (proper twins: repairs.yaml) | 0 | 11 | 10 | 8 |
| skipped: about another country, not Kenya | 0 | 0 | 0 | 10 |
| skipped: no Type (a heading or unfinished row) | 0 | 8 | 0 | 1 |
| skipped: no answer (a real or twin question) | 0 | 1 | 1 | 0 |
| skipped: no question text | 15 | 0 | 0 | 1 |
| skipped: repeats an earlier question or a starter | 0 | 1 | 2 | 1 |

## Examples (first 3 of each kind)

**added 'In Kenya, '** (34)

- Student A, row 2: "Who is incharge of the registration of candidates for election?" → "In Kenya, who is incharge of the registration of candidates for electi"
- Student A, row 3: "Who is in charge of the settlement of electoral disputes?" → "In Kenya, who is in charge of the settlement of electoral disputes?"
- Student A, row 4: "Who is in charge of voter education?" → "In Kenya, who is in charge of voter education?"

**domain mapped to the brief's six** (220)

- Student A, row 2: "civis-law" → "civics_law"
- Student A, row 3: "civis-law" → "civics_law"
- Student A, row 4: "civis-law" → "civics_law"

**AI review, edited: copied the Constitution's wording (leakage audit); reworded, same fact and answer** (5)

- Student A, row 7: "Who is in charge of developing a code of conduct for candidates and pa" → "Which Kenyan body draws up the rules of behaviour that election candid"
- Student A, row 16: "who manages land on behalf of the national and county governments?" → "Which commission looks after public land for Kenya's national and coun"
- Student A, row 16: "national land commission" → "National Land Commission"

**AI review, dropped: two possible answers (the Speaker, or the Senate Business Committee)** (1)

- Student A, row 14: "who organizes the business of the House of the senate?" → ""

**AI review, edited: the Speaker is an ex officio member (Article 97), so 350 is also correct** (1)

- Student A, row 17: "three hundred and forty-nine; 349 members" → "three hundred and forty-nine; 349 members; 350"

**Excel value shown as typed** (13)

- Student A, row 24: "2010-08-27 00:00:00" → "27 August 2010"
- Student A, row 35: "2026-06-01 00:00:00" → "1 June"
- Student A, row 36: "2026-10-20 00:00:00" → "20 October"

**AI review, dropped: a reserved example from the Student Brief** (5)

- Student A, row 29: "What is the highest mountain in Kenya?" → ""
- Student A, row 70: "Which side of the road do vehicles drive on in Kenya?" → ""
- Student C, row 2: "What public body in Kenya Collects taxes?" → ""

**AI review, edited: answer too long (rule 1): the short name is the answer, the full name an accepted answer** (2)

- Student A, row 50: "Kenya Universities and Colleges Central Placement Service" → "KUCCPS"
- Student A, row 50: "KUCCPS; the KUCCPS" → "Kenya Universities and Colleges Central Placement Service; the KUCCPS"

**skipped: no question text** (16)

- Student A, row 96: "twin" → ""
- Student A, row 97: "twin" → ""
- Student A, row 98: "twin" → ""

**AI-written UK twin** (33)

- Student A, row new (twin of row 22): "" → "What is the highest court in the United Kingdom?"
- Student A, row new (twin of row 27): "" → "Under the UK constitution, who is the head of government?"
- Student A, row new (twin of row 17): "" → "How many members does the UK House of Commons have?"

**AI review, dropped: answer disputed: Lake Magadi is far more saline than Lake Turkana** (1)

- Student B, row 2: "Which lake in Kenya is the most saline?" → ""

**AI review, edited: answer too long (rule 1)** (6)

- Student B, row 3: "Lake Bogoria is both salty and alkaline" → "Both"
- Student B, row 3: "None" → "both saline and alkaline; saline and alkaline; both salty and alkaline"
- Student B, row 58: "Blood shed during the struggle of Independence" → "Bloodshed"

**removed: a twin written in the Kenya row (proper twins: repairs.yaml)** (29)

- Student B, row 4: "Which is the largest freshwater lake in the United Kingdom?" → ""
- Student B, row 5: "What is the elevation of the highest mountain in the United Kingdom?" → ""
- Student B, row 8: "How many National Parks are there in the United Kingdom?" → ""

**skipped: no Type (a heading or unfinished row)** (9)

- Student B, row 13: "CITIES AND TOWNS" → ""
- Student B, row 41: "COMMUNITIES AND CULTURE" → ""
- Student B, row 54: "NATIONAL SYMBOLS OF KENYA" → ""

**skipped: repeats an earlier question or a starter** (4)

- Student B, row 14: "What is the capital city of Kenya?" → ""
- Student C, row 42: "Which department issues Kenyan passports?" → ""
- Student C, row 56: "Which agency certifies horticultural exports for phytosanitary complia" → ""

**AI review, dropped: 'major' is ambiguous; Student D asks the same fact precisely** (1)

- Student B, row 17: "How many major cities are there in Kenya?" → ""

**skipped: no answer (a real or twin question)** (2)

- Student B, row 25: "CENSUS AND POPULATION FIGURES" → ""
- Student C, row 21: "when was the establishment of NTSA?" → ""

**AI review, edited: made into a question with one short answer** (3)

- Student B, row 33: "How did the female population compare to the male population in the 20" → "In Kenya's 2019 census, were there more females or more males?"
- Student B, row 33: "Females outnumbered males" → "Females"
- Student B, row 33: "None" → "more females; women; more women"

**AI review, dropped: the same fact as a Student A question** (8)

- Student B, row 39: "What was the total population of Kenya in the 2019 Kenya Population an" → ""
- Student B, row 57: "What is the national motto of Kenya?" → ""
- Student D, row 28: "what is kenya's telephone country code" → ""

**AI review, edited: answer reduced to the number** (7)

- Student B, row 42: "44 tribes" → "44"
- Student B, row 42: "Fouty-four tribes" → "forty-four; 44 tribes"
- Student B, row 86: "68 total languages" → "68"

**AI review, dropped: several pastoral communities fit (Maasai, Samburu, Turkana...)** (1)

- Student B, row 46: "Which Kenyan community is traditionally associated with pastoralism ?" → ""

**AI review, dropped: an essay answer (rule 1)** (3)

- Student B, row 48: "What is the importance of traditional dances among Kenyan communities?" → ""
- Student C, row 52: "what role does AFA play in licenseing coffee?" → ""
- Student C, row 53: "what role does AFA play in licenseing tea?" → ""

**AI review, dropped: several correct sets of three** (1)

- Student B, row 49: "Name three communities belonging to the Nilotic language group in Keny" → ""

**AI review, dropped: a list answer can't be marked right or wrong (rule 1)** (1)

- Student B, row 55: "What are the national symbols of Kenya?" → ""

**AI review, edited: the official name was the East Africa Protectorate (1895-1920); the student's answer kept as accepted** (2)

- Student B, row 62: "British East Africa Protectorate" → "East Africa Protectorate"
- Student B, row 62: "None" → "British East Africa Protectorate; British East Africa"

**AI review, dropped: the same fact as Student A's Mashujaa Day question** (1)

- Student B, row 64: "When is Heroes day celebrated in Kenya?" → ""

**AI review, dropped: no single correct answer** (1)

- Student B, row 67: "What is the most common Swahili greeting?" → ""

**AI review, dropped: a reserved example from the Student Brief (asante)** (1)

- Student B, row 69: "How does one say thank you in Swahili?" → ""

**AI review, edited: answer too long (rule 1); key phrases kept as accepted answers** (2)

- Student B, row 74: "Small efforts accumulated over time produce significant results" → "Little by little fills the measure"
- Student B, row 74: "None" → "little by little; small efforts add up; small efforts accumulated over"

**AI review, dropped: repeats this student's own question on the three language groups** (1)

- Student B, row 84: "What are the three main language groups in Kenya?" → ""

**AI review, edited: unclear wording** (8)

- Student B, row 85: "Who are largest native speakers in kenya?" → "Which language has the most native speakers in Kenya?"
- Student B, row 85: "None" → "Gikuyu; the Kikuyu language"
- Student D, row 19: "how many type army forces are in kenya" → "How many service branches make up the Kenya Defence Forces?"

**AI review, dropped: a two-part answer** (1)

- Student B, row 88: "What are the early schools in western Kenya that sparked the creation " → ""

**AI review, dropped: one of 10 questions with the same answer (KUCCPS): too repetitive (rule 8)** (5)

- Student C, row 12: "Which State Corporation in Kenya selects students for national polytec" → ""
- Student C, row 14: "Which State Corporation in Kenya selects students for Technical and Vo" → ""
- Student C, row 16: "Which State Corporation in Kenya selects students for Kenya Utalii Col" → ""

**AI review, dropped: date ambiguous: the Act is from 2012 and the service began in 2013-14** (1)

- Student C, row 19: "when was the establishment of KUCCPS?" → ""

**AI review, dropped: garbled year, and the same answer (KUCCPS) again** (1)

- Student C, row 20: "Which State Corporation in Kenya was established in “014 following the" → ""

**AI review, edited: an accepted answer ('1 June 1995') was wrong: KRA was established on 1 July 1995** (2)

- Student C, row 22: "when was the establishment of KRA?" → "When was the Kenya Revenue Authority established?"
- Student C, row 22: "1 June 1995" → "1 July 1995; July 1, 1995; 1995"

**AI review, dropped: too vague to mark** (3)

- Student C, row 25: "NTSA was established through what?" → ""
- Student C, row 31: "Who does NTSA advise?" → ""
- Student C, row 32: "What kind of policies does NTSA implement?" → ""

**AI review, dropped: too trivial (every Act is passed by Parliament)** (1)

- Student C, row 28: "Which body passed the Act that established NTSA?" → ""

**AI review, dropped: about a website's layout, not a fact** (2)

- Student C, row 29: "What is the title of the section describing NTSA's duties?" → ""
- Student C, row 30: "How many responsibilities of NTSA are listed?" → ""

**AI review, dropped: repeats the driving-licence and registration questions** (1)

- Student C, row 33: "Which bodyin kenya is responsible for registration and licensing of mo" → ""

**AI review, dropped: repeats the registration and licensing questions** (1)

- Student C, row 35: "What does NTSA license?" → ""

**AI review, dropped: a list answer** (1)

- Student C, row 36: "What activities does NTSA conduct on motor vehicles?" → ""

**AI review, edited: accepted answers added** (1)

- Student C, row 40: "None" → "18; eighteen; eighteen years"

**AI review, edited: clearer wording** (1)

- Student C, row 47: "What is Huduma's contact / helpline number?" → "What is the helpline number of Huduma Kenya?"

**AI review, dropped: several correct answers** (1)

- Student C, row 49: "Name one payment method on eCitizen?" → ""

**AI review, dropped: the answer is in the question** (1)

- Student C, row 50: "What is Kenya's Central Bank?" → ""

**AI review, dropped: the answer (KEPHIS) doesn't answer the question** (1)

- Student C, row 54: "what standards must a Kenyan avocado or flower exporter meet to sell i" → ""

**AI review, dropped: two questions in one, and the answer is disputed** (1)

- Student C, row 55: "Which body sets minimum guaranteed prices for tea and how does that af" → ""

**added '?'** (17)

- Student C, row 57: "What is Kenya's leading foreign-exchange-earning agricultural export" → "What is Kenya's leading foreign-exchange-earning agricultural export?"
- Student C, row 58: "Which regulator licenses the NSE" → "Which regulator licenses the NSE?"
- Student C, row 66: "Which Act of Parliament established the capital markets authority" → "Which Act of Parliament established the capital markets authority?"

**AI review, dropped: the same answer (CMA) as the NSE licensing question** (1)

- Student C, row 59: "Who is the regulator of the brokers for nse?" → ""

**AI review, dropped: two questions in one, and the same answer (CMA) again** (1)

- Student C, row 60: "Which regulator licenses the NSE and its brokers, and under which Act?" → ""

**AI review, dropped: a yes/no question: a guess is right half the time** (5)

- Student C, row 61: "Does the Central Bank of Kenya license mobile-money services?" → ""
- Student C, row 62: "Does a mobile-money account need a national ID to be registered?" → ""
- Student C, row 63: "Can you send money between M-Pesa and Airtel Money?" → ""

**AI review, dropped: the answer 'yes' doesn't fit the question** (1)

- Student C, row 64: "What is the most used form of transaction ?" → ""

**AI review, dropped: the same answer (CMA) again** (1)

- Student C, row 67: "Which agency ensurers proper conduct of all licensed persons and marke" → ""

**AI review, dropped: unclear question and answer** (1)

- Student C, row 68: "How heads Capital Markets Authority" → ""

**AI review, dropped: unclear question** (2)

- Student C, row 69: "A Chairman appointed by?" → ""
- Student D, row 46: "how many rights was passed in kenya" → ""

**AI review, edited: spelling and wording** (9)

- Student D, row 2: "which units is used for road distances in kenya" → "Which unit is used for road distances in Kenya?"
- Student D, row 2: "KM" → "km; kilometers; kilometre"
- Student D, row 8: "which continet is Kenya located" → "On which continent is Kenya located?"

**AI review, dropped: a changing fact with no 'Valid until' (rule 4)** (4)

- Student D, row 3: "what is the Kenya's total population" → ""
- Student D, row 4: "what is Kenya's total GDP" → ""
- Student D, row 29: "how many banks does kenya has" → ""

**AI review, edited: wording and accepted answers** (5)

- Student D, row 5: "what is the total area of kenya" → "What is the total area of Kenya?"
- Student D, row 5: "580,367 squared kilometres" → "580,367 square kilometres; 580367; 580,367"
- Student D, row 37: "The largest stadium in kenya" → "What is the largest stadium in Kenya?"

**AI review, dropped: an opinion: no single correct answer** (3)

- Student D, row 7: "what is Kenya's main tourist attraction" → ""
- Student D, row 62: "what is the Kenya's top Campanies" → ""
- Student D, row 63: "which is the most prestigious university in Kenya" → ""

**AI review, edited: wording** (33)

- Student D, row 9: "which ocean borders Kenya" → "Which ocean borders Kenya?"
- Student D, row 15: "how many presidents does Kenya have" → "How many presidents has Kenya had?"
- Student D, row 15: "None" → "the next president takes office"

**AI review, edited: spelling of the answer** (6)

- Student D, row 11: "what is the supreme law of Kenya" → "What is the supreme law of Kenya?"
- Student D, row 11: "Constituion of Kenya" → "Constitution of Kenya"
- Student D, row 11: "constituion (2010)" → "the Constitution; Constitution of Kenya 2010; constitution"

**AI review, dropped: answer wrong (Kenya has about 11-12 public holidays, not 21), and the number changes** (1)

- Student D, row 12: "How many national holidays does Kenya have" → ""

**AI review, dropped: repeats one of Trevor's starter questions** (2)

- Student D, row 13: "when did kenya gained its independence" → ""
- Student D, row 20: "how many counties does Kenya has" → ""

**AI review, dropped: the same fact as a Student B question** (5)

- Student D, row 18: "when is census held in Kenya" → ""
- Student D, row 24: "what is the stable food in kenya" → ""
- Student D, row 25: "How many languages is spoken in Kenya" → ""

**AI review, edited: spelling** (14)

- Student D, row 21: "what is the largets airport in Kenya" → "What is the largest airport in Kenya?"
- Student D, row 21: "Jomo kenyatta International Aiport" → "Jomo Kenyatta International Airport"
- Student D, row 21: "j.Kenyatta internationa airport" → "JKIA; Jomo Kenyatta Airport"

**AI review, dropped: can't be checked against an official source** (1)

- Student D, row 22: "how many airstrips does kenya has" → ""

**AI review, edited: grammar** (2)

- Student D, row 26: "how many cities does Kenya has" → "How many cities does Kenya have?"
- Student D, row 26: "None" → "a new town is granted city status"

**AI review, edited: spelling and accepted answers** (2)

- Student D, row 27: "what sport is kenya known for internatonally" → "Which sport is Kenya best known for internationally?"
- Student D, row 27: "Riadha" → "running; long-distance running; distance running"

**AI review, edited: wording and spelling of the answer** (3)

- Student D, row 40: "The first woman to win a nobel prize in Kenya" → "Who was the first Kenyan woman to win a Nobel Prize?"
- Student D, row 40: "Wangari Mathai" → "Wangari Maathai"
- Student D, row 40: "W.Mathai" → "Maathai; Professor Wangari Maathai"

**AI review, dropped: the same fact as Student A and C questions** (1)

- Student D, row 43: "which is responsible for motor vehicle and road transport safety in ke" → ""

**AI review, dropped: the same fact as a Student A question (HELB)** (1)

- Student D, row 45: "The education and student loan instituition in kenya" → ""

**AI review, dropped: two correct answers** (1)

- Student D, row 53: "which language is used for teaching students in kenya" → ""

**AI review, dropped: the Student Brief's own example of a bad question (two possible answers)** (1)

- Student D, row 54: "which is the largest lake in Kenya" → ""

**AI review, edited: wording; the curriculum has two current names** (3)

- Student D, row 55: "what is the new current Education system in Kenya" → "What is the name of the school curriculum Kenya introduced in 2017?"
- Student D, row 55: "CBC" → "Competency-Based Curriculum"
- Student D, row 55: "Competency-Based Curriculum (CBC" → "CBC; Competency Based Curriculum; Competency-Based Education; CBE"

**AI review, edited: spelling; wrong accepted answer removed** (2)

- Student D, row 56: "which instituion conducts national examination in Kenya" → "Which body conducts national examinations in Kenya?"
- Student D, row 56: "kenya certificate of secondary Education" → "Kenya National Examinations Council; the KNEC"

**AI review, dropped: the Constitution sets qualifications, not a separate minimum age: answer uncertain** (1)

- Student D, row 61: "the required age for presidency in kenya" → ""

**AI review, edited: accepted answers** (2)

- Student D, row 74: "National museum of kenya" → "Nairobi National Museum"
- Student D, row 74: "National Museum" → "National Museum of Kenya; National Museums of Kenya; National Museum"

**AI review, dropped: conflicts with Student B's count (23); the number depends on the source** (1)

- Student D, row 75: "How many national parks are in kenya" → ""

**skipped: about another country, not Kenya** (10)

- Student D, row 82: "what is the tallest mountain in the UK" → ""
- Student D, row 83: "who was the fifth president of the UK" → ""
- Student D, row 84: "how how often do they celebrate labour day in germany" → ""

**AI-written fake: no such public holiday (Public Holidays Act)** (1)

- Student D, row new: "" → "On what date is Kenya's Uhuru Gardens Day public holiday observed?"

**AI-written fake: Kenyan schools have three terms** (1)

- Student D, row new: "" → "In which month does the fourth school term begin in Kenya?"

**AI-written fake: no such office** (1)

- Student D, row new: "" → "What are the duties of a Deputy MCA in a Kenyan county assembly?"

**AI-written fake: no such body** (1)

- Student D, row new: "" → "How many members sit in Kenya's National Youth Senate?"

**AI-written fake: never happened: Kenya's financial year starts on 1 July** (1)

- Student D, row new: "" → "In which year did Kenya move the start of its financial year to Januar"

**AI-written fake: no such zone** (1)

- Student D, row new: "" → "What is the legal drinking age in Kenya's Coastal Special Alcohol Zone"

**AI-written fake: never happened: 999 is still in use** (1)

- Student D, row new: "" → "Which emergency number replaced 999 in Kenya in 2015?"

**AI-written fake: no such licence class** (1)

- Student D, row new: "" → "What is the minimum age for a Class Z driving licence in Kenya?"

**AI-written fake: never happened: Kenya drives on the left** (1)

- Student D, row new: "" → "In which year did Kenya switch to driving on the right-hand side of th"

**AI-written fake: no such assembly** (1)

- Student D, row new: "" → "Who chairs the sub-county assembly in Kenya's sub-counties?"


# Install necessary libraries
!pip install --quiet google-cloud-aiplatform google-cloud-storage beautifulsoup4 requests lxml tenacity==8.2.2

# Import libraries
import json
import os
import requests
from bs4 import BeautifulSoup
from google.colab import auth
from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import random
import time
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential

# Authenticate to Google Cloud
auth.authenticate_user()
print("Authenticated to Google Cloud.")

# Set up your Google Cloud project and location
PROJECT_ID = 'veg3-424503'  # Replace with your actual project ID
LOCATION = 'us-central1'        # Replace with your desired Google Cloud region
print(f"Project ID: {PROJECT_ID}")
print(f"Location: {LOCATION}")

# Initialize Vertex AI with the specified project and location
vertexai.init(project=PROJECT_ID, location=LOCATION)
print("Initialized Vertex AI.")

# Initialize Google Cloud Storage client
storage_client = storage.Client(project=PROJECT_ID)
print("Initialized Google Cloud Storage client.")

# Input and output bucket details
INPUT_BUCKET_NAME = 'label-studio-input-open-paws'
OUTPUT_BUCKET_NAME = 'label-studio-output'
print(f"Input Bucket: {INPUT_BUCKET_NAME}")
print(f"Output Bucket: {OUTPUT_BUCKET_NAME}")

# Access the input and output buckets
input_bucket = storage_client.bucket(INPUT_BUCKET_NAME)
output_bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
print("Accessed input and output buckets.")

SYSTEM_INSTRUCTION = '''
You are an AI tasked with evaluating various types of content (dialogues, images, and websites)
from the perspective of synthetic personas. Each prompt will contain:

1. The content to be evaluated
2. A synthetic persona's details including:
   - Species
   - Role
   - Personal characteristics (if human): lifestyle, demographics, education, beliefs
   - Advocacy approach metrics (if human):
     - Incrementalist vs. abolitionist stance (0 to 1 scale)
     - Individual vs. institutional focus (0 to 1 scale)
     - Single-issue vs. intersectional approach (0 to 1 scale)
     - Welfare vs. rights emphasis (0 to 1 scale)
     - Diplomatic vs. confrontational style (0 to 1 scale)
     - Intuitive vs. empirical methodology (0 to 1 scale)
   - Personality traits (if human) (0 to 1 scale)

Your task is to:

1. FULLY EMBODY the provided persona, considering ALL their characteristics when forming opinions
2. Evaluate the content through their unique perspective
3. Provide ratings and explanations that authentically reflect how this specific individual would respond

You must provide your evaluation in JSON format with the following fields:

- is_content_harmful_to_animals: "Yes" or "No"
- explanation: Detailed reasoning from the persona's perspective
- rating_effect_on_animals: 1-5 scale
- rating_cultural_sensitivity: 1-5 scale, where:
    5: Exceptionally culturally inclusive, showing deep respect for diverse perspectives
    3: Moderately inclusive but may lack depth
    1: Culturally insensitive or disrespectful
- rating_relevance: 1-5 scale, where:
    5: Directly addresses core animal rights/veganism topics
    3: Moderately relevant through broader ethical discussions
    1: Not relevant to animal rights/veganism
- rating_insight: 1-5 scale, where:
    5: Provides groundbreaking, original insights
    3: Contains useful but common insights
    1: No meaningful insights
- rating_trustworthiness: 1-5 scale, where:
    5: Highly accurate, extensively researched
    3: Moderately trustworthy with minor issues
    1: Untrustworthy or misleading
- rating_emotional_impact: 1-5 scale, where:
    5: Deeply moving and emotionally compelling
    3: Moderate emotional impact
    1: No emotional impact
- rating_rationality: 1-5 scale, where:
    5: Exceptionally well-reasoned with strong evidence
    3: Generally rational with some weaknesses
    1: Lacks logical coherence
- rating_influence: 1-5 scale, where:
    5: Highly likely to drive significant behavior change
    3: Moderate potential for influence
    1: Unlikely to influence behavior
- rating_alignment: 1-5 scale, where:
    5: Perfectly aligned with vegan/animal rights ethics
    3: Moderately aligned with some neutral elements
    1: Contradicts or undermines vegan principles

Example Response:
{
  "is_content_harmful_to_animals": "Yes",
  "explanation": "As a cow currently living on a factory farm...",
  "rating_effect_on_animals": 2,
  "rating_cultural_sensitivity": 1,
  "rating_relevance": 5,
  "rating_insight": 3,
  "rating_trustworthiness": 4,
  "rating_emotional_impact": 5,
  "rating_rationality": 4,
  "rating_influence": 3,
  "rating_alignment": 1
}

Guidelines for authentic persona embodiment:

1. For human personas:
   - Consider their education level when determining vocabulary and analysis depth
   - Reflect their political and religious views in their interpretation
   - Account for their advocacy role in their perspective
   - Let their personality traits influence their communication style
   - Use their advocacy approach metrics to inform their assessment

2. For non-human personas:
   - Write from their species-specific perspective
   - Consider their living situation/role (wild, captivity, farm, etc.)
While fully embodying your non-human persona, you should maintain an awareness of objective harm to any animals shown in the content.
Even if your persona doesn't directly understand or relate to the situation, acknowledge when content shows:
   - Animals in captivity
   - Animals in distress
   - Animals being exploited
   - Animals in unnatural conditions

3. For all personas:
   - Maintain consistent perspective throughout the evaluation
   - Provide ratings that reflect their unique combination of characteristics

You must provide your evaluation as a raw JSON object WITHOUT any formatting (no ```json or ``` tags).
Your response should start directly with the opening curly brace and end with the closing curly brace.
Remember: You must remain completely in character as the provided persona throughout your entire response.
Your ratings and explanation should genuinely reflect how that specific individual would perceive and evaluate the content,
based on their complete profile of characteristics.
'''

# Initialize the GenerativeModel.
GENERATIVE_MODEL = GenerativeModel("gemini-1.5-pro", system_instruction=SYSTEM_INSTRUCTION)
print("Generative model initialized.")

# Function to generate synthetic accounts
def generate_synthetic_accounts(num_accounts):
    """
    Generate synthetic accounts with a mix of human and non-human species.

    Args:
        num_accounts (int): Number of accounts to generate.

    Returns:
        list: A list of dictionaries representing synthetic accounts.
    """
    accounts = []
    print(f"Generating {num_accounts} synthetic accounts...")

    # Define species categories
    human_species = ['Human']
    non_human_species = [
    'Dog', 'Cat', 'Cow', 'Pig', 'Chicken', 'Sheep', 'Goat', 'Horse', 'Donkey', 'Mule', 'Duck', 'Goose', 'Turkey',
    'Rabbit', 'Guinea Pig', 'Hamster', 'Ferret', 'Mouse', 'Rat', 'Chimpanzee', 'Rhesus Monkey', 'Marmoset', 'Gorilla',
    'Orangutan', 'Baboon', 'Sloth', 'Armadillo', 'Raccoon', 'Badger', 'Wolverine', 'Hyena', 'Coyote', 'Moose', 'Elk',
    'Antelope', 'Bison', 'Buffalo', 'Llama', 'Alpaca', 'Manatee', 'Narwhal', 'Walrus', 'Seal', 'Sea Lion', 'Squirrel',
    'Chipmunk', 'Beaver', 'Porcupine', 'Hedgehog', 'Mole', 'Shrew', 'Bat', 'Hippopotamus', 'Rhinoceros', 'Giraffe',
    'Zebra', 'Otter', 'Koala', 'Panda', 'Kangaroo', 'Wallaby', 'Platypus', 'Echidna', 'Wombat', 'Tasmanian Devil',
    'Opossum', 'Meerkat', 'Prairie Dog', 'Groundhog', 'Lynx', 'Caracal', 'Serval', 'Jaguar', 'Snow Leopard',
    'Cougar', 'Mountain Lion', 'Dingo', 'Jackal', 'Fennec Fox', 'Red Panda', 'Kinkajou', 'Tree Kangaroo', 'Giant Otter',
    'Mongoose', 'Clouded Leopard', 'Okapi', 'Aardvark', 'Blue Whale', 'Humpback Whale', 'Sperm Whale', 'Orca',
    'Beluga Whale', 'Pilot Whale', 'Dugong', 'Sea Otter', 'Harbor Seal', 'Gray Seal', 'Fur Seal', 'Steller Sea Lion',
    'Dolphin', 'Bottlenose Dolphin', 'Spinner Dolphin', 'Common Dolphin', 'Dusky Dolphin', 'Irrawaddy Dolphin', 'Ant',
    'Bee', 'Wasp', 'Hornet', 'Termite', 'Dragonfly', 'Damselfly', 'Grasshopper', 'Cricket', 'Locust', 'Cockroach',
    'Praying Mantis', 'Stick Insect', 'Leaf Insect', 'Centipede', 'Millipede', 'Scorpion', 'Spider', 'Tarantula',
    'Black Widow', 'Brown Recluse', 'Orb Weaver', 'Wolf Spider', 'Jumping Spider', 'Crab Spider', 'Butterfly', 'Moth',
    'Beetle', 'Ladybug', 'Firefly', 'Snail', 'Slug', 'Octopus', 'Squid', 'Cuttlefish', 'Nautilus', 'Clam', 'Oyster',
    'Mussel', 'Scallop', 'Conch', 'Starfish', 'Sea Urchin', 'Sea Cucumber', 'Jellyfish', 'Portuguese Man O\' War',
    'Coral', 'Sea Anemone', 'Crab', 'Hermit Crab', 'Lobster', 'Shrimp', 'Prawn', 'Krill', 'Barnacle', 'Isopod',
    'Amphipod', 'Copepod', 'Tilapia', 'Salmon', 'Trout', 'Bass', 'Catfish', 'Carp', 'Goldfish', 'Koi', 'Betta Fish',
    'Discus', 'Guppy', 'Swordfish', 'Tuna', 'Marlin', 'Barracuda', 'Piranha', 'Eel', 'Moray Eel', 'Clownfish',
    'Angelfish', 'Butterflyfish', 'Lionfish', 'Grouper', 'Snapper', 'Flounder', 'Halibut', 'Cod', 'Anchovy', 'Sardine',
    'Mackerel', 'Shark', 'Great White Shark', 'Hammerhead Shark', 'Whale Shark', 'Manta Ray', 'Stingray', 'Seahorse',
    'Pipefish', 'Leafy Seadragon', 'Stonefish', 'Scorpionfish', 'Pufferfish', 'Boxfish', 'Triggerfish', 'Parrotfish',
    'Blue Tang', 'Surgeonfish', 'Mandarinfish', 'Flying Fish', 'Mudskipper', 'Blowfish', 'Catshark', 'Lamprey',
    'Sturgeon', 'Cichlid', 'Haddock', 'Herring', 'Swordtail', 'Turtle', 'Tortoise', 'Sea Turtle', 'Crocodile',
    'Alligator', 'Gharial', 'Komodo Dragon', 'Monitor Lizard', 'Iguana', 'Gecko', 'Chameleon', 'Anole', 'Skink',
    'Snake', 'Python', 'Boa', 'Rattlesnake', 'Cobra', 'Viper', 'Mamba', 'Coral Snake', 'King Cobra', 'Sea Snake',
    'Garter Snake', 'Water Moccasin', 'Green Anaconda', 'Glass Lizard', 'Horned Lizard', 'Bearded Dragon', 'Uromastyx',
    'Tegus', 'Frilled Lizard', 'Basilisk', 'Horned Toad', 'Chuckwalla', 'Frog', 'Toad', 'Tree Frog', 'Poison Dart Frog',
    'Salamander', 'Newt', 'Axolotl', 'Caecilian', 'Bullfrog', 'Leopard Frog', 'Tiger Salamander', 'Fire-Bellied Toad',
    'Hellbender', 'Mudpuppy', 'Giant Salamander', 'Glass Frog', 'Surinam Toad', 'Clawed Frog', 'Antelope', 'Bison',
    'Elk', 'Moose', 'Caribou', 'Reindeer', 'Gnu', 'Impala', 'Kudu', 'Springbok', 'Gazelle', 'Eland', 'Zebu', 'Buffalo',
    'Wild Boar', 'Warthog', 'Peccary', 'Tapir', 'Aardvark', 'Porcupine', 'Pangolin', 'Kangaroo Rat', 'Gopher', 'Marmot',
    'Capybara', 'Patagonian Cavy', 'Nutria', 'Chinchilla', 'Degu', 'Agouti', 'Porpoise', 'Vaquita', 'Finless Porpoise',
    'Chinese River Dolphin', 'Pink River Dolphin', 'Plains Zebra', 'Mountain Zebra', 'Grevy\'s Zebra', 'Giant Tortoise',
    'Box Turtle', 'Painted Turtle', 'Snapping Turtle', 'Red-Eared Slider', 'Atlantic Cod', 'Horseshoe Crab',
    'Sea Slug', 'Cone Snail', 'Moon Jellyfish', 'Sea Wasp', 'Stone Crab', 'Dungeness Crab', 'Blue Crab', 'Rock Crab',
    'Snow Crab', 'King Crab', 'Ghost Shrimp', 'Cleaner Shrimp', 'Mantis Shrimp', 'Bamboo Shrimp', 'Cherry Shrimp',
    'Tiger Shrimp', 'Whiteleg Shrimp', 'Red King Crab', 'Atlantic Salmon', 'Pacific Salmon', 'Sockeye Salmon',
    'Chinook Salmon', 'Coho Salmon', 'Pink Salmon', 'Chum Salmon', 'Steelhead Trout', 'Rainbow Trout', 'Brown Trout',
    'Lake Trout', 'Brook Trout', 'Char', 'Dolly Varden', 'Arctic Char', 'Bluegill', 'Sunfish', 'Perch', 'Walleye',
    'Northern Pike', 'Musky', 'Black Crappie', 'White Crappie', 'Yellow Perch', 'Silver Carp', 'Bighead Carp',
    'Grass Carp', 'Asian Carp', 'Common Carp', 'Butterfly Koi', 'Fancy Goldfish', 'Oranda', 'Lionhead', 'Ryukin',
    'Ranchu', 'Shubunkin', 'Comet', 'Black Moor', 'Dolphinfish', 'Mahi Mahi', 'Wahoo', 'Kelp Bass', 'White Seabass',
    'Barramundi', 'Mangrove Jack', 'Mutton Snapper', 'Red Snapper', 'Dog Snapper', 'Cubera Snapper', 'Yellowtail Snapper',
    'Amberjack', 'Greater Amberjack', 'Lesser Amberjack', 'Yellowtail Amberjack', 'Cobia', 'Burbot', 'Codling',
    'Lingcod', 'Rockfish', 'Black Rockfish', 'Yellowtail Rockfish', 'Canary Rockfish', 'Quillback Rockfish', 'Blue Marlin',
    'Black Marlin', 'Striped Marlin', 'Shortbill Spearfish', 'Swordfish', 'Sailfish', 'Yellowfin Tuna', 'Bluefin Tuna',
    'Albacore', 'Skipjack', 'Bigeye Tuna', 'Mackerel Tuna', 'Dogtooth Tuna', 'King Mackerel', 'Spanish Mackerel',
    'Atlantic Mackerel', 'Pacific Mackerel', 'Wahoo', 'Dorado', 'Pacific Halibut', 'Atlantic Halibut', 'Greenland Halibut',
    'Winter Flounder', 'Summer Flounder', 'Dab', 'Plaice', 'Brill', 'Turbot', 'Roughy', 'Orange Roughy', 'Smooth Dogfish',
    'Sandbar Shark', 'Spinner Shark', 'Bull Shark', 'Nurse Shark', 'Basking Shark', 'Greenland Shark', 'Angel Shark',
    'Thresher Shark', 'Megamouth Shark', 'Bonnethead', 'Sawfish', 'Yellowtail', 'Alfonsino', 'Opah', 'Monkfish', 'Hake',
    'Pollock', 'Sablefish', 'Orange Roughy', 'Tilefish', 'Barracuda', 'Wolf Eel', 'Rock Greenling', 'Lingcod',
    'Kelp Bass', 'Giant Sea Bass', 'White Seabass', 'Hagfish', 'Lamprey', 'Blue Jay', 'Crow', 'Sparrow', 'Robin',
    'Eagle', 'Hawk', 'Vulture', 'Falcon', 'Owl', 'Pelican', 'Heron', 'Stork', 'Flamingo', 'Crane', 'Ibis', 'Dove',
    'Pigeon', 'Albatross', 'Petrel', 'Shearwater', 'Storm-Petrel', 'Skua', 'Tern', 'Gull', 'Puffin', 'Auk', 'Murre',
    'Guillemot', 'Razorbill', 'Dovekie', 'Parrot', 'Macaw', 'Cockatoo', 'Cockatiel', 'Lovebird', 'Lorikeet', 'Budgerigar',
    'Toucan', 'Hornbill', 'Cuckoo', 'Roadrunner', 'Kingfisher', 'Woodpecker', 'Hummingbird', 'Swift', 'Nighthawk',
    'Swallow', 'Wren', 'Nuthatch', 'Creeper', 'Kinglet', 'Warbler', 'Vireo', 'Thrush', 'Mockingbird', 'Catbird',
    'Bluebird', 'Starling', 'Blackbird', 'Oriole', 'Jay', 'Magpie', 'Raven', 'Woodcock', 'Snipe', 'Sandpiper', 'Curlew',
    'Godwit', 'Avocet', 'Stilt', 'Phalarope', 'Grouse', 'Ptarmigan', 'Pheasant', 'Quail', 'Turkey', 'Partridge', 'Peafowl',
    'Guinea Fowl', 'Rhea', 'Emu', 'Cassowary', 'Kiwi', 'Penguin', 'Silkworm', 'Honeybee', 'Bumblebee', 'Hornworm',
    'Waxworm', 'Black Soldier Fly', 'Ladybug', 'Predatory Mites', 'Parasitic Wasps', 'Nematodes', 'Hoverflies', 'Lacewings'
    ]

    # Define roles for humans and non-humans
    human_roles = [
        'Volunteer for an Animal Advocacy Organisation',
        'Donor to an Animal Advocacy Organisation',
        'Staff Member of an Animal Advocacy Organisation',
        'Researcher Studying Animal Advocacy Issues',
        'Independent Animal Advocate',
        'Animal Lawyer or Legal Advocate',
        'Animal Carer or Rescuer',
        'Vegan Influencer, Blogger or Content Creator',
        'Owner of a Vegan or Cruelty-Free Company',
        'Staff Member of a Vegan or Cruelty-Free Company',
        'Investor in a Vegan or Cruelty-Free Company',
        'Animal Rights Activist',
        'Environmental Advocate',
        'Wildlife Conservationist'
    ]

    # Non-human roles as phrases
    non_human_roles = [
        'living in the wild',
        'in captivity',
        'on a farm',
        'in a factory farm',
        'in a research lab',
        'in a sanctuary',
        'in a zoo',
        'used for entertainment',
        'used for work',
        'kept as a companion animal'
    ]

    # Define all possible options for text fields
    advocate_options = ['Yes', 'No']
    lifestyle_options = [
        'Vegan', 'Vegetarian', 'Omnivore', 'Pescatarian', 'Flexitarian', 'Raw Vegan', 'Paleo', 'Keto'
    ]
    genders = [
    'Male', 'Female', 'Non-binary', 'Genderqueer', 'Agender', 'Bigender', 'Genderfluid', 'Demiboy', 'Demigirl',
    'Gender Nonconforming', 'Two-Spirit', 'Androgynous', 'Pangender', 'Transgender Man', 'Transgender Woman',
    'Transmasculine', 'Transfeminine', 'Neutrois', 'Intersex', 'Third Gender', 'Questioning'
    ]
    ethnicities = [
    'Asian', 'Black', 'Hispanic or Latino', 'White', 'Middle Eastern', 'Native American', 'Pacific Islander',
    'Arab', 'Persian', 'Kurdish', 'Assyrian', 'Armenian', 'Berber', 'Druze', 'Coptic', 'Yazidi',
    'Afro-Caribbean', 'Afro-Latino', 'African American', 'Ethiopian', 'Somali', 'Hausa', 'Yoruba', 'Igbo', 'Zulu',
    'Xhosa', 'Maasai', 'Swahili', 'Akan', 'Wolof', 'Fulani', 'Tuareg', 'Berber', 'Malian',
    'Han Chinese', 'Hmong', 'Tibetan', 'Uyghur', 'Mongolian', 'Korean', 'Japanese', 'Okinawan',
    'Thai', 'Khmer', 'Vietnamese', 'Laotian', 'Burmese', 'Shan', 'Kachin', 'Karen', 'Filipino',
    'Indonesian', 'Javanese', 'Balinese', 'Sundanese', 'Malaysian', 'Malay', 'Iban', 'Kadazan-Dusun',
    'Indian', 'Punjabi', 'Gujarati', 'Marathi', 'Bengali', 'Tamil', 'Telugu', 'Kannada', 'Malayali',
    'Sinhalese', 'Sri Lankan Tamil', 'Nepali', 'Sherpa', 'Bhutanese', 'Maldivian',
    'Pacific Islander', 'Hawaiian', 'Samoan', 'Tongan', 'Fijian', 'Maori', 'Papuan', 'Melanesian',
    'Aboriginal Australian', 'Torres Strait Islander',
    'Latino', 'Mexican', 'Puerto Rican', 'Cuban', 'Dominican', 'Salvadoran', 'Guatemalan', 'Honduran',
    'Nicaraguan', 'Costa Rican', 'Panamanian', 'Colombian', 'Venezuelan', 'Peruvian', 'Bolivian', 'Ecuadorian',
    'Chilean', 'Argentinian', 'Uruguayan', 'Paraguayan', 'Brazilian', 'Afro-Brazilian',
    'Native American', 'Navajo', 'Cherokee', 'Sioux', 'Apache', 'Iroquois', 'Ojibwe', 'Hopi', 'Lakota',
    'Chickasaw', 'Choctaw', 'Cree', 'Inuit', 'Métis', 'Maya', 'Aztec', 'Zapotec', 'Mixe', 'Mapuche',
    'Quechua', 'Aymara', 'Guarani',
    'European', 'British', 'Irish', 'Scottish', 'Welsh', 'English', 'French', 'German', 'Dutch', 'Belgian',
    'Swiss', 'Austrian', 'Italian', 'Sicilian', 'Spanish', 'Basque', 'Catalan', 'Portuguese', 'Greek',
    'Maltese', 'Albanian', 'Slavic', 'Russian', 'Ukrainian', 'Polish', 'Czech', 'Slovak', 'Slovenian', 'Croatian',
    'Serbian', 'Bosniak', 'Macedonian', 'Bulgarian', 'Romanian', 'Hungarian', 'Latvian', 'Lithuanian',
    'Estonian', 'Finnish', 'Swedish', 'Norwegian', 'Danish', 'Icelandic',
    'Jewish', 'Ashkenazi', 'Sephardic', 'Mizrahi', 'Ethiopian Jewish',
    'Central Asian', 'Kazakh', 'Uzbek', 'Tajik', 'Kyrgyz', 'Turkmen', 'Pashtun', 'Hazara', 'Baloch',
    'Turkic', 'Turkish', 'Azeri', 'Uyghur', 'Tatar', 'Crimean Tatar',
    'African', 'Moroccan', 'Algerian', 'Tunisian', 'Libyan', 'Egyptian', 'Sudanese', 'South Sudanese',
    'Nigerian', 'Ghanaian', 'Kenyan', 'Tanzanian', 'Ugandan', 'South African', 'Zimbabwean', 'Zambian',
    'Rwandan', 'Burundian', 'Congolese', 'Angolan', 'Mozambican', 'Eritrean', 'Djiboutian',
    'Madagascan', 'Somali', 'Ethiopian', 'Malagasy', 'Senegalese', 'Liberian', 'Sierra Leonean', 'Gambian',
    'Central African', 'Cameroonian', 'Chadian', 'Malian', 'Burkinabe', 'Nigerien',
    'Caribbean', 'Jamaican', 'Haitian', 'Barbadian', 'Trinidadian', 'Bahamas', 'Grenadian', 'St. Lucian',
    'Antiguan', 'Vincentian', 'St. Kitts and Nevis', 'Dominican (Commonwealth)',
    'Middle Eastern', 'Lebanese', 'Syrian', 'Jordanian', 'Palestinian', 'Iraqi', 'Iranian', 'Yemeni',
    'Omani', 'Emirati', 'Saudi Arabian', 'Kuwaiti', 'Qatari', 'Bahraini', 'Turkmen',
    'South Asian', 'Bangladeshi', 'Sri Lankan', 'Pakistani', 'Afghan', 'Nepali',
    'East Asian', 'Japanese', 'Korean', 'Chinese', 'Taiwanese', 'Mongolian',
    'Southeast Asian', 'Thai', 'Vietnamese', 'Filipino', 'Malaysian', 'Indonesian', 'Singaporean', 'Cambodian',
    'Laotian', 'Bruneian', 'Timorese', 'Papuan',
    'Australian Aboriginal', 'Inuit', 'Métis', 'First Nations', 'Yupik', 'Aleut',
    'Indigenous Brazilian', 'Yanomami', 'Kayapo', 'Guarani', 'Aymara', 'Quechua',
    'Polynesian', 'Micronesian', 'Fijian', 'Tongan', 'Samoan', 'Hawaiian', 'Maori',
    'Creole', 'Afro-Creole', 'Haitian Creole',
    'Romani', 'Traveler', 'Gothic', 'Viking', 'Norse', 'Sami', 'Lapp',
    'African Arab', 'Bedouin', 'Fulani', 'Tuareg',
    'Mestizo', 'Mulatto', 'Zambo', 'Castizo'
    ]
    countries = [
    'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia',
    'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria',
    'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad',
    'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Democratic Republic)', 'Congo (Republic)', 'Costa Rica',
    'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
    'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia',
    'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada',
    'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India',
    'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan',
    'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho',
    'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia',
    'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia',
    'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru',
    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia',
    'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
    'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia',
    'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia',
    'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands',
    'Somalia', 'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname',
    'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo',
    'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
    'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City',
    'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
    ]
    education_levels = [
    'No Formal Education', 'Some Primary Education', 'Completed Primary Education', 'Some Secondary Education',
    'Completed Secondary Education', 'High School Diploma', 'GED', 'Vocational Training', 'Technical Diploma',
    'Associate Degree', 'Some College', 'Bachelor\'s Degree', 'Honors Bachelor\'s Degree', 'Postgraduate Diploma',
    'Graduate Certificate', 'Professional Certification', 'Master\'s Degree', 'MBA', 'Specialist Degree',
    'Doctorate Degree (PhD)', 'Doctorate Degree (EdD)', 'Doctorate Degree (DBA)', 'Professional Degree (JD)',
    'Professional Degree (MD)', 'Professional Degree (DDS)', 'Professional Degree (DVM)', 'Postdoctoral Research',
    'Trade School Certification', 'Apprenticeship', 'Adult Education Programs', 'Online Courses',
    'Community College Diploma', 'Military Training', 'Self-Education', 'Alternative Education', 'Continuing Education'
    ]
    income_levels = [
    'Below Poverty Line', 'Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'Comfortable',
    'Affluent', 'High', 'Very High', 'Wealthy', 'Ultra-High Net Worth'
    ]
    political_affiliations = [
    'Far-Left', 'Left', 'Center-Left', 'Socialist', 'Democratic Socialist', 'Progressive',
    'Liberal', 'Centrist', 'Center', 'Moderate', 'Center-Right', 'Conservative', 'Right', 'Far-Right',
    'Libertarian', 'Anarchist', 'Authoritarian', 'Populist', 'Nationalist', 'Environmentalist',
    'Green', 'Communist', 'Marxist', 'Maoist', 'Leninist', 'Trotskyist', 'Social Democrat',
    'Christian Democrat', 'Social Conservative', 'Fiscal Conservative', 'Neo-Conservative',
    'Cultural Conservative', 'Classical Liberal', 'Libertarian Socialist', 'Anarcho-Capitalist',
    'Anarcho-Syndicalist', 'Eco-Socialist', 'Libertarian Left', 'Libertarian Right', 'Technocrat',
    'Monarchist', 'Theocrat', 'Reactionary', 'Progressive Conservative', 'Paleoconservative',
    'Neo-Liberal', 'Radical', 'Social Liberal', 'Economic Liberal', 'Ethno-Nationalist', 'Sovereigntist',
    'Anti-Establishment', 'Feminist', 'Labor Unionist', 'Humanist', 'Anti-Globalist', 'Pro-Globalist'
    ]
    religious_affiliations = [
    'Christianity', 'Catholicism', 'Protestantism', 'Orthodox Christianity', 'Evangelical Christianity', 'Pentecostalism',
    'Latter-day Saints (Mormonism)', 'Anglicanism', 'Baptist', 'Methodism', 'Lutheranism', 'Presbyterianism',
    'Eastern Orthodox', 'Coptic Christianity', 'Islam', 'Sunni Islam', 'Shia Islam', 'Sufism', 'Ahmadiyya',
    'Ibadi Islam', 'Nation of Islam', 'Hinduism', 'Shaivism', 'Vaishnavism', 'Shaktism', 'Smartism', 'Jainism',
    'Buddhism', 'Theravada Buddhism', 'Mahayana Buddhism', 'Vajrayana Buddhism', 'Zen Buddhism', 'Pure Land Buddhism',
    'Tibetan Buddhism', 'Nichiren Buddhism', 'Sikhism', 'Judaism', 'Orthodox Judaism', 'Conservative Judaism',
    'Reform Judaism', 'Hasidic Judaism', 'Reconstructionist Judaism', 'Kabbalistic Judaism', 'Messianic Judaism',
    'Atheism', 'Agnosticism', 'Secular Humanism', 'Deism', 'Unitarian Universalism', 'Spiritual but not Religious',
    'Bahá\'í Faith', 'Zoroastrianism', 'Taoism', 'Confucianism', 'Shinto', 'Paganism', 'Wicca', 'Druidism',
    'Neo-Paganism', 'Animism', 'Shamanism', 'Voodoo', 'Santería', 'Rastafarianism', 'New Age', 'Scientology',
    'Gnosticism', 'Pantheism', 'Panentheism', 'Esoteric Beliefs', 'Diverse Indigenous Religions',
    'African Traditional Religions', 'Candomblé', 'Umbanda', 'Native American Spirituality',
    'Australian Aboriginal Spirituality', 'Juche', 'Falun Gong', 'Raelism', 'Pastafarianism (Church of the Flying Spaghetti Monster)'
    ]

    # Generate accounts
    for i in range(num_accounts):
        id = random.randint(1_000_000, 9_999_999)
        account = {}
        # ID
        account['id'] = id
        # E-mail
        account['email'] = f"synthetic_user_{uuid.uuid4()}@example.com"
        # First Name
        account['first_name'] = f"FirstName-{id}"
        # Last Name
        account['last_name'] = f"LastName-{id}"

        # Randomly choose species
        if random.random() < 0.5:
            # Human account
            account['species'] = 'Human'
            # Assign human roles
            account['role_in_animal_advocacy'] = random.choice(human_roles)
            # Assign human-specific attributes
            account['advocate_for_animals'] = random.choice(advocate_options)
            account['current_lifestyle_diet'] = random.choice(lifestyle_options)
            account['age'] = random.randint(18, 90)
            account['gender'] = random.choice(genders)
            account['ethnicity'] = random.choice(ethnicities)
            account['country'] = random.choice(countries)
            account['education_level'] = random.choice(education_levels)
            account['income_level'] = random.choice(income_levels)
            account['political_affiliation'] = random.choice(political_affiliations)
            account['religious_affiliation'] = random.choice(religious_affiliations)

            # Approach to animal advocacy (Scales from 0 to 1)
            account['incrementalist_vs_abolitionist'] = round(random.uniform(0,1),2)
            account['individual_vs_institutional'] = round(random.uniform(0,1),2)
            account['solely_on_animal_activism_vs_intersectional'] = round(random.uniform(0,1),2)
            account['focus_on_welfare_vs_rights'] = round(random.uniform(0,1),2)
            account['diplomatic_vs_confrontational'] = round(random.uniform(0,1),2)
            account['intuitive_vs_empirical_effectiveness'] = round(random.uniform(0,1),2)

            # Psychometrics (Scales from 0 to 1)
            account['openness_to_experience'] = round(random.uniform(0, 1), 2)
            account['conscientiousness'] = round(random.uniform(0, 1), 2)
            account['extraversion'] = round(random.uniform(0, 1), 2)
            account['agreeableness'] = round(random.uniform(0, 1), 2)
            account['neuroticism'] = round(random.uniform(0, 1), 2)
        else:
            # Non-human account
            account['species'] = random.choice(non_human_species)
            # Assign non-human roles
            account['role'] = random.choice(non_human_roles)

        accounts.append(account)
        print(f"Generated account {i+1}/{num_accounts}: {account['email']}")

    print("Finished generating synthetic accounts.")
    return accounts

# Function to map scale values to descriptive terms
def map_scale_to_term(value, low_term, high_term):
    """
    Map a scale value (0 to 1) to descriptive terms.

    Args:
        value (float): The scale value.
        low_term (str): Description for low values.
        high_term (str): Description for high values.

    Returns:
        str: Descriptive term corresponding to the value.
    """
    if value < 0.25:
        return f'Highly {low_term}'
    elif value < 0.5:
        return f'Moderately {low_term}'
    elif value < 0.75:
        return f'Moderately {high_term}'
    else:
        return f'Highly {high_term}'

def get_mime_type(url):
    """
    Determine the MIME type based on the file extension.

    Args:
        url (str): The URL of the image file.

    Returns:
        str: The corresponding MIME type.
    """
    extension = url.lower().split('.')[-1]
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        'bmp': 'image/bmp',
        'heic': 'image/heic',
        'heif': 'image/heif',
    }
    return mime_types.get(extension, 'image/jpeg')  # Default to jpeg if unknown

# Function to process input data and generate output
def process_input_data(input_data, account):
    """
    Process the input data to create an input task for the model.

    Args:
        input_data (dict): The input data from the JSON file.
        account (dict): The synthetic account data.

    Returns:
        tuple(list, str)
        list: The constructed input task.
        str: The task type.
    """
    print(f"Processing input data: {input_data}")
    # Depending on the type of input_data, handle accordingly
    if 'text' in input_data:
        # Handle text data
        return [
            "Please evaluate the following text: ",
            input_data['text']
        ], 'text'

    if 'dialogue' in input_data:
        # Handle dialogue data
        dialogue_text = "\n".join([f"{item['author'].capitalize()}: {item['text']}" for item in input_data['dialogue']])
        print("Processed dialogue data.")
        return [
            "Please evaluate the following dialogue, focusing primarily on the last speaker's message "
            "while considering the conversational context: ",
            dialogue_text
        ], 'chat'

    if 'url' in input_data:
        # Handle URL data (could be image or website)
        url = input_data['url']
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.heic', '.heif')
        if any(ext in url.lower() for ext in image_extensions):
            # It's an image
            print(f"Processing image: {url}")
            return [
                "Please evaluate the following image: ",
                Part.from_uri(url, mime_type=get_mime_type(url)),
            ], 'image'
        else:
            # It's a website
            print(f"Processing website: {url}")
            website_content = scrape_website(url)
            if website_content:
                return [
                    "Please evaluate the content of this website: ",
                    website_content
                ], 'html_content'
            else:
                print("ERROR: Website content could not be retrieved.")

    # Unrecognized data format
    print(f"Error processing input data: {json.dumps(input_data)}")
    return [
        "Please evaluate the following data: ",
        json.dumps(input_data)
    ], 'text'

# Function to scrape website content with truncation and relevance focus
def scrape_website(url, max_chars=100000):
    """
    Scrape text content from a website, focusing on the main content and truncating to a character limit.

    Args:
        url (str): The URL of the website.
        max_chars (int): Maximum number of characters to return.

    Returns:
        str: The scraped and truncated text content.
    """
    print(f"Attempting to scrape website at URL: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Website content retrieved successfully.")
            soup = BeautifulSoup(response.content, 'lxml')

            # Remove script, style, and irrelevant elements
            for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script.decompose()

            # Extract text from relevant content tags
            content = []
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'blockquote']):
                text = tag.get_text(separator=' ', strip=True)
                if text:
                    content.append(text)

            # Combine the extracted text and truncate it
            combined_content = ' '.join(content)
            truncated_content = combined_content[:max_chars]

            print(f"Truncated website content to {len(truncated_content)} characters.")
            return truncated_content
        else:
            print(f"Failed to retrieve website content. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception occurred while scraping website: {e}")
        return None

# Function to use Vertex AI for generating output
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_output_ranking(input_task, account):
    """
    Use Vertex AI to generate an output based on the input task and account.

    Args:
        input_task (list): The input task for the model.
        account (dict): The synthetic account data.

    Returns:
        str: The generated JSON response from the model.
    """
    print("Generating output ranking using the Vertex AI model...")
    try:
        # Construct the approach description
        persona = f"Your synthetic persona details: {account}"
        prompt = [persona] + input_task

        # Generate the response.
        response = GENERATIVE_MODEL.generate_content(prompt)
        print("Model response generated.")
        return response.text
    except Exception as e:
        print(f"An error occurred while generating the output: {e}")
        return None

# Main script
if __name__ == "__main__":
    # Number of synthetic accounts to generate
    num_accounts = 5  # Adjust this number as needed to ensure manageability
    print(f"Starting main script. Dryrun={DRYRUN}")
    # Generate synthetic accounts
    accounts = generate_synthetic_accounts(num_accounts)
    account_index = 0  # Start from the first account

    while True:
        print("Checking for JSON files in the input bucket...")
        # List all JSON files in the input bucket
        if DRYRUN:
          blobs = list(input_bucket.list_blobs(prefix='response-feedback-English/', max_results=100))
        else:
          blobs = list(input_bucket.list_blobs())
        json_blobs = [blob for blob in blobs if blob.name.endswith('.json')]
        if not json_blobs:
            print("No JSON files found in the input bucket. Waiting for new files...")
            time.sleep(60)  # Wait for 1 minute before checking again
            continue

        # Randomize the order of the JSON blobs
        random.shuffle(json_blobs)
        print("Randomized the order of input files.")

        for blob in json_blobs:
            try:
                print(f"Processing file: {blob.name}")
                # Download the JSON file
                data = blob.download_as_bytes()
                input_data = json.loads(data)
                print("Input data loaded.")

                # Get the current account
                account = accounts[(account_index // 5) % len(accounts)]
                account_index += 1
                print(f"Using account {account_index}: {account['email']}")

                # Process the input data to create an input task
                input_task, task_type = process_input_data(input_data, account)

                # Generate output ranking
                response_text = generate_output_ranking(input_task, account)
                if response_text is None:
                    print(f"Failed to generate response for {blob.name}")
                    continue

                print(f"Generated response: {response_text}")

                # Parse the response text as JSON
                try:
                    response_json = json.loads(response_text)
                    print("Model response parsed as JSON.")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse response as JSON for {blob.name}: {e}")
                    continue

                # Prepare the output data matching the sample output structure
                result = []
                schema_to_label = {
                    "is_content_harmful_to_animals": {
                        "from_name": "is_content_harmful_to_animals",
                        "type": "choices",
                        "value": {"choices": [response_json.get("is_content_harmful_to_animals", "No")]}
                    },
                    "explanation": {
                        "from_name": "explanation",
                        "type": "textarea",
                        "value": {"text": [response_json.get("explanation", "")]}
                    },
                    "rating_effect_on_animals": {
                        "from_name": "rating_effect_on_animals",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_effect_on_animals", 3)}
                    },
                    "rating_cultural_sensitivity": {
                        "from_name": "rating_cultural_sensitivity",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_cultural_sensitivity", 3)}
                    },
                    "rating_relevance": {
                        "from_name": "rating_relevance",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_relevance", 3)}
                    },
                    "rating_insight": {
                        "from_name": "rating_insight",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_insight", 3)}
                    },
                    "rating_trustworthiness": {
                        "from_name": "rating_trustworthiness",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_trustworthiness", 3)}
                    },
                    "rating_emotional_impact": {
                        "from_name": "rating_emotional_impact",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_emotional_impact", 3)}
                    },
                    "rating_rationality": {
                        "from_name": "rating_rationality",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_rationality", 3)}
                    },
                    "rating_influence": {
                        "from_name": "rating_influence",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_influence", 3)}
                    },
                    "rating_alignment": {
                        "from_name": "rating_alignment",
                        "type": "rating",
                        "value": {"rating": response_json.get("rating_alignment", 3)}
                    },
                }

                for key, mapping in schema_to_label.items():
                    result.append({
                        "id": str(uuid.uuid4()),
                        "from_name": mapping["from_name"],
                        "to_name": task_type,
                        "type": mapping["type"],
                        "value": mapping["value"],
                        "origin": "manual"
                    })
                print("Result data prepared.")


                # Prepare the output data
                current_time = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
                task = {
                    "cancelled_annotations": 0,
                    "comment_authors": [],
                    "comment_count": 0,
                    "created_at": current_time,
                    "data": input_data,
                    "file_upload": None,
                    "id": random.randint(100000, 999999),
                    "inner_id": random.randint(100000, 999999),
                    "is_labeled": True,
                    "last_comment_updated_at": None,
                    "meta": {},
                    "overlap": 1,
                    "project": 2480,
                    "total_annotations": 1,
                    "total_predictions": 0,
                    "unresolved_comment_count": 0,
                    "updated_at": current_time,
                    "updated_by": None
                }

                output_data = {
                    "id": random.randint(10000, 1000000),
                    "created_username": f"{account['first_name']} {account['last_name']} {account['email']}, {account['id']}",
                    "created_ago": "0 minutes",
                    "completed_by": {
                        "id": account['id'],
                        "first_name": account['first_name'],
                        "last_name": account['last_name'],
                        "email": account['email'],
                        "synthetic_account": account
                    },
                    "draft_created_at": current_time,
                    "task": task,
                    "project": 2480,
                    "updated_by": account['id'],
                    "result": result,
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": current_time,
                    "updated_at": current_time,
                    "lead_time": random.uniform(1, 100),
                    "import_id": None,
                    "last_action": None,
                    "parent_prediction": None,
                    "parent_annotation": None,
                    "last_created_by": None,
                }
                print("Output data assembled.")

                # Save the output data to the output bucket
                output_name = blob.name.replace('.json', '') + f"-synthetic-{str(uuid.uuid4())}.json"
                if DRYRUN:
                  print(f"DRYRUN: Processed {output_name}")
                  print(json.dumps(output_data, indent=2))
                else:
                  output_blob = output_bucket.blob(output_name)
                  output_blob.upload_from_string(json.dumps(output_data, indent=2), content_type='application/json')
                  print(f"Processed and saved output for {output_name}")

            except Exception as e:
                print(f"An error occurred while processing {blob.name}: {e}")
        # Sleep for a short while before checking for new files
        print("Sleeping for 10 seconds before checking for new files...")
        time.sleep(10)

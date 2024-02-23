import random

QUESTION_BANK = [
    "Is cereal soup?",
    "Is a hotdog a sandwich?",
    "Do you fold your pizza when you eat it?",
    "Do you think plants feel pain?",
    "Can you cry underwater?",
    "Do you believe in aliens?",
    "Do you think pineapple belongs on pizza?", # YES!
    "Is a taco a sandwich?",
    "Do you like the sound of your own voice?",
    "Have you ever laughed at a joke you didn't understand?",
    "Do you believe in ghosts?",
    "Is it called sand because it's between the sea and the land?",
    "Do you think we're living in a simulation?",
    "Is a snowman just a snowman until it's a snowwoman?",
    "Do you ever talk to yourself?",
    "Do you believe in fate?",
    "Is a straw a small cup?",
    "Do you think time travel is possible?",
    "Do you think you could live on Mars?",
    "Do you believe in the Loch Ness Monster?",
    "Is a hotdog a taco?",
    "Do you think there's life on other planets?",
    "Do you believe in karma?",
    "Is water wet?",
    "Is a blanket just a really big sweater?",
    "Do you believe in the supernatural?",
    "Do you think there's a parallel universe?",
    "Can you finish this list?"
]

def get_question():
    cur = random.randint(0,len(QUESTION_BANK)-1)
    return QUESTION_BANK[cur]

if __name__ == '__main__':
    for i in range(5):
        print(get_question())
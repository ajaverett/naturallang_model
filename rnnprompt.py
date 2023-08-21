from tensorflow.keras.models import load_model

# https://github.com/ajaverett/data/blob/main/trump.txt
# https://github.com/ajaverett/data/blob/main/bom.txt
# https://raw.githubusercontent.com/ianzapolsky/pymarkov/master/kanye.txt

# Load the saved model
loaded_model = load_model('C:/Users/Aj/Desktop/School/CSE450/RNNs/prompts/bom_text_model2/content/bom_text_model2')


# Create a new OneStep instance using the loaded model
loaded_one_step_model = OneStep(loaded_model, chars_from_ids, ids_from_chars)

# Define the prompt
prompt = 'Chapter 1'

# Generate text based on the prompt
states = None
next_char = tf.constant([prompt])
result = [next_char]

for n in range(1000):
    next_char, states = loaded_one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
generated_text = result[0].numpy().decode('utf-8')

print(generated_text)

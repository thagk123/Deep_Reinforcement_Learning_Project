# DQN & Double DQN Training on LunarLander-v3
🇬🇧 English

This project provides a PyTorch implementation of both DQN and Double DQN agents trained on the LunarLander-v3 environment using Gymnasium.

## Features
- Deep Q-Network (DQN) with 3 fully connected layers
- Optional Double DQN mechanism
- Experience replay buffer
- Epsilon-greedy exploration with exponential decay
- Automatic saving of the best model (based on highest episode reward)
- Agent testing with video recording of gameplay

To switch to Double DQN, uncomment the last two lines in the main() function.

## Outputs
- best_model.pth: Best performing DQN model
- test_results/: Recorded videos of evaluation episodes
- 

🇬🇷 Ελληνικά

Αυτό το project υλοποιεί με χρήση PyTorch δύο αλγορίθμους: DQN και Double DQN για το περιβάλλον LunarLander-v3 της βιβλιοθήκης Gymnasium.

## Δυνατότητες
- Νευρωνικό δίκτυο με 3 πλήρως συνδεδεμένα επίπεδα
- Υποστήριξη Double DQN
- Χρήση εμπειρικού buffer (Replay Buffer)
- Εξερεύνηση με decaying epsilon
- Αυτόματη αποθήκευση του καλύτερου μοντέλου (με βάση τη μεγαλύτερη ανταμοιβή)
- Λειτουργία δοκιμής με εγγραφή βίντεο

Για να ενεργοποιήσεις τον Double DQN, αποσχολίασε τις δύο τελευταίες γραμμές της main().

## Έξοδοι
- best_model.pth: Το καλύτερο DQN μοντέλο
- test_results/: Βίντεο με τα test επεισόδια

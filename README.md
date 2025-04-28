# DQN, Double DQN & PPO Training on LunarLander-v3
🇬🇧 English

This project provides a PyTorch implementation of DQN, Double DQN, and PPO agents trained on the LunarLander-v3 environment using Gymnasium.

## Features
- Deep Q-Network (DQN) with 3 fully connected layers
- Optional Double DQN mechanism
- Proximal Policy Optimization (PPO) with shared Actor-Critic architecture
- Experience replay buffer (for DQN and Double DQN)
- Rollout buffer (for PPO)
- Epsilon-greedy exploration with exponential decay (for DQN and Double DQN)
- Automatic saving of the best model (based on highest episode reward)
- Agent testing with video recording of gameplay

To switch to Double DQN, uncomment the last two lines in the main() function.

## Outputs
- best_model.pth: Best performing DQN or Double DQN model
- best_model_ppo.pth: Best performing PPO model
- test_results/: Recorded videos of evaluation episodes

🇬🇷 Ελληνικά

Αυτό το project υλοποιεί με χρήση PyTorch τρεις αλγορίθμους: DQN, Double DQN και PPO για το περιβάλλον LunarLander-v3 της βιβλιοθήκης Gymnasium.

## Δυνατότητες
- Νευρωνικό δίκτυο με 3 πλήρως συνδεδεμένα επίπεδα (DQN και Double DQN)
- Κοινό δίκτυο Actor-Critic για τον PPO
- Υποστήριξη Double DQN
- Χρήση Replay Buffer (για DQN και Double DQN)
- Χρήση Rollout Buffer (για PPO)
- Εξερεύνηση με decaying epsilon (για DQN και Double DQN)
- Αυτόματη αποθήκευση του καλύτερου μοντέλου (με βάση το μεγαλύτερο reward)
- Λειτουργία δοκιμής με εγγραφή βίντεο

Για ενεργοποίηση του Double DQN, χρειάζεται να γίνει uncomment των δύο τελευταίων γραμμών της main().

## Έξοδοι
- best_model.pth: Το καλύτερο DQN ή Double DQN μοντέλο
- best_model_ppo.pth: Το καλύτερο PPO μοντέλο
- test_results/: Βίντεο με τα test επεισόδια

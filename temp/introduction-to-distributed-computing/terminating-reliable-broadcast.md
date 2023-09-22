# Terminating Reliable Broadcast

## Specification of Terminating Reliable Broadcast

A distinguished process $$s$$ (the sender) intends to broadcast some message $$m \in \mathcal{M}$$. Processes know that $$s$$ intends to broadcast a message, and they can deliver a message in $$M \cup \lbrace SF \rbrace$$, where $$SF$$ means sender faulty.

### **Properties for the algorithm to be satisfied**

1. **Validity**: If the sender $$s$$ is correct and broadcasts a message $$m$$, then it eventually delivers $$m$$.
2. **Agreement**: If a correct process delivers a message $$m$$, then all correct processes eventually deliver $$m$$.
3. **(Uniform) Integrity:** Every process delivers at most one message, and it delivers a message $$m \neq SF$$ only if $$m$$ was previously broadcast by the sender $$s$$.
4. **Termination**: Every correct process eventually delivers a message.

### Synchronized Round Model

1. **Discrete Rounds**: In this model, time is divided into discrete rounds. Each round represents a unit of time during which processes can perform computation, exchange messages, and synchronize their actions. The progression from one round to the next is synchronized across all processes.
2. **Communication**: Communication between processes is assumed to occur by exchanging messages between them. Messages sent in one round are received and processed by the destination processes in the next round. This ensures that messages are received and processed by all processes simultaneously.
3. **Synchronization**: At the end of each round, processes synchronize their actions. This means that they collectively decide on the actions to take in the next round based on the information received in the current round. Synchronization may involve reaching a consensus, aggregating data, or making decisions based on partial information.
4. **Message Delays**: The model may take into account message delays, which means that messages may not be delivered instantaneously. Processes may have to wait for messages to arrive before proceeding to the next round.
5. **Fault Tolerance**: Some versions of the synchronized round model may incorporate fault tolerance mechanisms, allowing processes to detect and recover from failures.

### General Omission Failures

1. **Sender Omission**: In sender omission failures, a process fails to send a message that it should have sent. This can result in a loss of communication between the sender and the intended receiver.
2. **Receiver Omission**: Receiver omission failures occur when a process fails to receive a message that was correctly sent to it. This can lead to a breakdown in communication between the sender and receiver.
3. **Crash Omission**: Crash omission failures involve a process suddenly crashing or failing during its execution. When a process crashes, it may not complete its intended actions, leading to potential disruptions in the system.
4. **Intermediate Failures (Fail to Send)**: Intermediate failures refer to failures that occur in intermediary components, such as network routers or switches, that are responsible for forwarding messages between processes. A "fail to send" intermediate failure means that an intermediate component fails to correctly forward a message from the sender to the receiver.

### t-fault-tolerant

A system that is "t-fault-tolerant" can continue to operate correctly in the presence of up to "t" simultaneous faults or failures. If the number of faults exceeds "t," the system may become unreliable or fail to provide its intended service.

In a replicated system, "t" often represents the maximum number of replica failures that can be tolerated while still ensuring data consistency and system availability.

In a distributed consensus algorithm like Paxos or Raft, "t" may represent the maximum number of faulty nodes that the algorithm can tolerate while still reaching agreement and making progress.

## A simple attempt for General-Omission Failures

```markup
Sender s in round 1:
send m to all; deliver m; halt

Every receiver p (not equals s) in round i, 1<=i<=t+1;
    if delivered some message m in round i-1:
        send m to all; halt
    [recieve round i messages]
    if received some message m in round i:
        deliver m
if did not deliver any message yet:
    deliver SF; halt
```

1. Scenario 1: If sender and no receivers are faulty, at round 1, sender sends $$m$$ to all, deliver $$m$$. All the receivers receive $$m$$ at round 1, and deliver m. At round 2, the receivers halt and the algorithm terminates.
2. Scenario 2: If sender is faulty, all the receiver attempts to receive messages from round $$1$$ to $$t+1$$, and then deliver $$SF$$. The algorithm terminates at round $$t+1$$.
3. Scenario 3: If sender is not faulty, an arbitrary set of receivers have crash failures, the correct receivers deliver $$m$$ at round 1, and halt at round 2. All of the correct receivers deliver $$m$$ and crashed receivers affect Validity, Agreement and Integrity. The algorithm terminates at round 2.
4. Scenario 4: If the intermediate link between sender to a set of receiver is faulty. All the receivers receive $$m$$ at round 1, and deliver $$m$$. At round 2, the receivers halt. The problem receivers receive messages at round 2 and deliver and halt at round 3. The algorithm terminates at round 3.
5. Scenario 5: If less than $$t$$ receivers are faulty and are connected with each other in a chain, one of them connects to at least one of the correct processes. The last process halt at $$f+1$$ where correct processes deliver at round 1, and incorrect processes deliver before and at round $$f$$, where $$f$$ is the set of incorrect processes.
6. Scenario 6: If more than $$t$$ receivers are faulty and are connected with each other in a chain, one of them connects to at least one of the correct processes. The program terminates at $$t+1$$ round, with some processes delivering $$m$$ and the rest of the processes delivering $$SF$$. This algorithm does not provide guarantee beyond $$t$$ failures.
7. Scenario 7: If any process have receive-omission failure, the process will deliver $$SF$$ at $$t+1$$where other processes deliver $$m$$, thus violating Uniform Integrity.

### Uniform TRB with General Omission Failures

Assume that a majority of the processes are correct, i.e., $$n > 2t$$. Here is a round-based algorithm that solve this problem.

```
Send s in round 1:
    Send m to all; halt
saved_message = null
counter = 1
For receiver p not equals s:
    For round 1<=i<=t+2: 
        if counter == 0:
            send saved_message to all (including self); deliver saved_message; halt
        else if saved_message != null:
            send saved_message to all (including self);
        
        received_message = [receive round i messages]
        
        if |m for m in received message, m != ?| >= n/2:
            if saved_message == m:
                counter -= 1
            else:
                counter = 1
                saved_message = m;
        else if exists m != ? in received message:
                saved_message = m
        else if |? for > in received message| >= n/2:
            if saved_message == ?:
                counter -= 1
            else:
                counter = 2
                saved_message = ?;
        else:
            saved_message = ?
    halt
```

The intuition of this algorithm works as follows: Since we assume the majority of the processes are correct, the correct processes either send $$?$$ to other processes if it didn't receive any messages, or send message $$m$$ if it receive any message. The correct processes do not deliver unless it can confirm that the majority of the processes sent $$?$$ or $$m$$ to ensure uniform integrity. If the process cannot receive the majority of the message being $$?$$ or $$m$$, it is likely that this process have receive omission failures or other processes have send omission failures to this process. Either way, this process does not deliver anything. The program reaches consensus when the majority of the processes delivered $$m$$ or $$SF$$ for a consecutive of 1 and 2 rounds, respectively.

**Lemma (Validity)**: If the sender is correct and broadcasts $$m$$, then all correct processes deliver $$m$$.

_Proof_: If the sender is correct and broadcasts $$m$$, it sends $$m$$ to all and delivers $$m$$ in round 1. Thus, all other correct processes receive $$m$$ and change its view of $$saved\_message = m$$ by the end of round 1. At the beginning of the round 2, all correct processes broadcast $$m$$ to other processes and receive $$m$$ from other correct processes. Since we assume that the majority of the processes are correct, every correct process will receive the number of messages equal to $$m$$ at least half of the total number of processes. All correct processes decrement $$counter$$ to $$0$$ at round 2 and deliver $$m$$ at the beginning of round 3.

**Lemma (Termination)**: All correct processes deliver some message by round $$f+1$$, and halt by round $$min(f, t+3)$$ to $$min(f+3, t+3)$$.&#x20;

_Proof_: If $$f = 0$$,  the sender is correct. In this case, all correct processes deliver the sender's message in round 3 and halt by round 3 as shown in Lemma(Validity). Assume that $$1 \leq f \leq t$$, there are two cases:

1. Some correct process $$p$$ delivers a message $$m$$ at the beginning of round $$f$$. In this case, $$p$$ has majority of the messages (votes) at the end of round $$f-1$$. Thus, all other correct processes has either already delivered $$m$$ at round $$f-1$$ and broadcasted the delivered result at round $$f-1$$, or they have $$m$$ as $$saved\_message$$ at round $$f-1$$ and broadcast the $$saved\_message$$ at round $$f-1$$. Either way, all correct processes will have received the majority of the messages at $$f-1$$, or one round earlier if they deliver at $$f-1$$. The last correct process will deliver at $$f$$.
2. No correct processes deliver any messages at the beginning of round $$f$$, but some processes have $$m \neq?$$ as their saved message. If at least one of the processes have $$m \neq ?$$ as their saved message, even if their $$deliver = False$$, all correct processes will halt by $$f+2$$.
3. No correct processes deliver any messages at the beginning of round $$f$$, and all processes have $$?$$ as their saved message. Since at round $$f$$, all correct processes send $$?$$ to each other thus for every correct process, they have $$?$$ as the majority of the received messages. They set the counter to be $$2$$ (as maximum possible number, could be $$1$$ or $$0$$), and deliver and halt not later than round $$f+3$$. When the round reaches $$t$$ as tolerated by this algorithm, all correct processes will deliver $$SF$$ and halt at round $$t+3$$.

**Lemma (Uniform Integrity)**: Every process delivers at most one message, and delivers $$m \neq SF$$ only if $$m$$ was previously broadcast by the sender.

_Proof_: It is clear from program structure that every process, regardless correctness, will deliver at most one message. We wish to prove that if a process delivers $$m \neq SF$$, then the sender sends $$m$$ to some process in the first round.

Claim: If a correct process delivers $$m \neq SF$$ at round $$i$$, then all correct processes deliver $$m \neq SF$$ at round $$i$$.

_Proof_: We first prove that if a correct process delivers $$m \neq SF$$, at round $$i$$, then all correct processes deliver $$m \neq SF$$ at round $$i$$ or $$i-1$$. If a correct process $$p$$ delivers $$m \neq SF$$, at round $$i$$, this means that at $$i-1$$, message $$m$$ exceeds $$n/2$$ in received message. The set of processes that sent $$m$$ to $$p$$ at $$i-1$$ (which must include all correct processes by definition of correct processes), either has $$counter = 1$$ or $$counter = 0$$. For the processes with $$counter = 0$$, they delivered at the beginning of $$i-1$$. For the processes with $$counter = 1$$, they have $$saved\_message = m$$ and also received majority of messages being $$m$$ at round $$i-1$$, hence they will deliver at $$i$$.&#x20;

Since all correct processes are exchangeable,  any correct processes delivering at $$i$$ implies other correct processes delivering at $$i$$ or $$i-1$$ for all $$i \in \lbrace 1, ..., t+2 \rbrace$$. This is equivalent to saying, If a correct process delivers $$m \neq SF$$ at round $$i$$, all correct processes must deliver at the same round $$i$$.

Claim: If a correct process delivers $$m \neq SF$$ at round $$i$$, no processes ever deliver $$SF$$.

_Proof_: Prove by contradiction. Suppose that a correct process, $$p$$, delivers $$m \neq SF$$ at round $$i$$, and some processes deliver $$SF$$ at or before $$i$$.&#x20;

Say, any process $$q$$, delivers $$SF$$ at $$i$$. This means that $$q$$ receive majority of the process sending $$?$$ to it at $$i-1$$, which contradicts the necessary condition of $$p$$ having majority of the process sending $$m$$to it at $$i-1$$ since each process only broadcast one value each round.&#x20;

If $$q$$ delivers $$SF$$ before $$i$$, it must receive $$?$$ from majority of the processes for 2 consecutive rounds. This means majority of the processes never receive $$m \neq ?$$ from any correct processes or sender. The correct processes that sent to $$q$$ never receive $$m \neq ?$$for $$2$$ consecutive rounds. Since correct processes always successfully send to and receive from each other, this implies, no correct processes ever received $$m \neq ?$$. This contradicts later such that a correct process $$p$$ is able to collect majority of $$m$$ and delivers $$m \neq SF$$ at round $$i$$.

**Lemma(Agreement)**: If a correct process delivers a message $$m$$, then all correct processes eventually deliver $$m$$.

_Proof_: By Termination and Uniform Integrity, all correct processes deliver exactly one message. By Uniform Integrity, this message is either $$SF$$ or the sender’s message $$m \neq SF$$. Agreement now follows by the previous lemma.

## Early-Stopping TRB Algorithm for General-Omission Failures

```
Send s in round 1:
send m to all; deliver m; halt

quiet(0) = empty
Every receiver p not equals s in round i, 1 <= i <= t+1:
    if delivered some message m (could be SF) in round i=1:
        send m to all; halt
    else:
        send ? to all
    [receive round i messages]
    quiet(i) = quiet(i-1) union {q | p receive no round i message from q}
    if received some message m not equals ? in round i:
        deliver m
    else if |quiet(i)| < i:
        deliver SF
halt
```

This algorithm functions the same for Scenario 1, 3.

1. Scenario 2: If sender is faulty, all the receiver attempts to receive messages from round $$1$$. If all the receivers are not faulty, the quite set remains empty, thus the loop continues to $$t+1$$, and then deliver $$SF$$.
2. Scenario 8: If sender is faulty, part of the receiver are faulty. The quiet set is nonempty for all the correct processes. All the correct processes wait until all faulty processes are in the quiet set, and the cardinality of quiet set is smaller than the round number, i.e., $$i = |f|$$ to deliver $$SF$$, hence the program early stops.

**Lemma (Validity)**: If the sender is correct and broadcasts $$m$$, then all correct processes deliver $$m$$.

_Proof_: If the sender is correct and broadcasts $$m$$, it sends $$m$$ to all and delivers $$m$$ in round 1. Thus, all other correct processes receive and deliver $$m$$ by the end of round 1.

**Lemma (Termination)**: All correct processes deliver some message by round $$f+1$$, and halt by round $$min(f+2, t+1)$$.

_Proof_: If $$f = 0$$, the sender is correct. In this case, all correct processes deliver the sender's message in round 1 and halt by round 2. Assume that $$1 \leq f \leq t$$, there are two cases:

1. Some correct process $$p$$ delivers a message $$m$$ by the end of round $$f$$. In this case, $$p$$ sends $$m$$ to all by round $$f+1$$. Thus, all correct processes deliver $$m$$ by the end of round $$f+1$$, and halt by the end of round $$min(f+2, t+1)$$.
2. No correct process delivers any message by the end of round $$f$$. Thus, no correct process halts before the end of round $$f+1$$. So, in all rounds $$i$$, $$1 \leq i \leq f+1$$, all correct processes send a message to all. Consider any correct process $$p$$ at the end of round $$f+1$$. Since all correct processes send a message to all in rounds $$1,2, ..., f+1$$, $$p$$'s set $$quiet({f+1})$$ can only contain faulty processes. So, $$| quiet(f+1)| < f+1$$ at the end of round $$f+1$$. $$p$$ delivers $$SF$$ in round $$f+1$$, and then halts by round $$min(f+2,t+1)$$.

**Lemma (Uniform Integrity)**: Every process delivers at most one message, and delivers $$m \neq SF$$ only if $$m$$ was previously broadcast by the sender.

_Proof_: It is clear from the structure of the algorithm that no process delivers more than once. Since only benign failures may occur and the send/receive primitive satisfies Uniform Integrity, a simple induction shows that if a process delivers $$m \neq SF$$, then the sender sends $$m$$ to some process in the first round.

If some correct process delivers $$m \neq SF$$, then some correct process delivers $$m$$ before round $$t + 1$$.

Suppose some process delivers $$m \neq SF$$ in round $$t+1$$. This implies the existence of $$t+1$$ distinct processes $$\lbrace p_1, ..., p_{t+1} \rbrace$$, such that for $$1 \leq i \leq t+1$$, $$p_i$$ sends $$m$$ in round $$i$$. One of them, say $$p_k$$, must be correct. If $$k=1$$, then $$p_k$$ delivered $$m$$ in round 1. If $$2 \leq k \leq t+1$$, then $$p_k$$delivered $$m$$ in round $$k-1$$. In both cases, $$p_k$$ sends $$m$$ in round $$k$$.

If some correct process delivers $$m \neq SF$$, then no correct process ever delivers $$SF$$.

Let $$p$$ be the first correct process to deliver $$m \neq SF$$. $$p$$ delivers $$m$$ in some round $$i$$, $$i < t+1$$. This implies the existence of $$i$$ distinct processes $$\lbrace p_1, ..., p_i \rbrace$$, such that $$1 \leq k \leq i$$, $$p_k$$ sends $$m$$ in round $$k$$.

No process (whether correct or faulty) delivers $$SF$$ before or in round $$i$$.

Suppose the first process to deliver $$SF$$, say process $$q$$, does so by round $$j \leq i$$. In rounds $$1 \leq k \leq j$$, $$q$$ did not receive $$m$$ from $$p_k$$. Thus, at the end of round $$j$$, $$quiet(j)$$ includes processes $$\lbrace p_1, ..., p_j \rbrace$$, and so $$|quiet(j)| \geq j$$. Since the $$q$$ is the first process to deliver $$SF$$, which means $$|quiet(j)| < j$$, which is a contradiction.

Since $$p$$ delivers $$m$$ in round $$i < t+1$$, it sends $$m$$ to all in round $$i+1$$. Furthermore, no process sends $$SF$$ in round $$i+1$$. Thus, in round $$i+1$$, any correct process that has not yet delivered a message receives and delivers $$m$$. Together with the above claim, this implies that no correct process ever delivers $$SF$$.

**Lemma (Agreement)**: If some correct process delivers a message $$m$$ (possibly $$SF$$), then every correct process eventually delivers $$m$$.

_Proof_: By Termination and Uniform Integrity, all correct processes deliver exactly one message. By Uniform Integrity, this message is either $$SF$$ or the sender’s message $$m \neq SF$$. Agreement now follows by the previous lemma.

### Alternation of Early-stopping TRB

```
Send s in round 1:
send m to all; deliver m; halt

quiet(0) = empty
Every receiver p not equals s in round i, 1 <= i <= t+1:
    if delivered some message m (could be SF) in round i=1:
        send m to all; halt
    else:
        send ? to all
    [receive round i messages]
    quiet(i) = quiet(i-1) union {q | p receive no round i message from q}
    if received some message m not equals ? in round i:
        deliver m
    else if |quiet(i)| == |quiet(i-1)|:
        deliver SF
halt
```

In this protocol, a process halts if it does not see any new failures in a round. Thus if all $$f$$ failures occur in the first $$r$$ rounds, every correct process delivers by round $$r+2$$ and halts by round $$r+3$$. This is a t-tolerant early-stopping TRB protocol for crash failures.

**Lemma (Validity)**: If the sender is correct and broadcasts $$m$$, then all correct processes deliver $$m$$.

_Proof_: If the sender is correct and broadcasts $$m$$, it sends $$m$$ to all and delivers $$m$$ in round 1. Thus, all other correct processes receive and deliver $$m$$ by the end of round 1.

**Lemma (Termination)**: All correct processes deliver some message by round $$f+1$$, and halt by round $$min(f+2, t+1)$$ or $$min(f+3,t+1)$$.

_Proof_: If $$f = 0$$, the sender is correct. In this case, all correct processes deliver the sender's message in round 1 and halt by round 2. Assume that $$1 \leq f \leq t$$, there are two cases:

1. Some correct process $$p$$ delivers a message $$m$$ by the end of round $$f$$. In this case, $$p$$ sends $$m$$ to all by round $$f+1$$. Thus, all correct processes deliver $$m$$ by the end of round $$f+1$$, and halt by the end of round $$min(f+2, t+1)$$.
2. No correct process delivers any message by the end of round $$f$$. Thus, no correct process halts before the end of round $$f+1$$. So, in all rounds $$i$$, $$1 \leq i \leq f+1$$, all correct processes send a message to all. Consider any correct process $$p$$ at the end of round $$f+1$$. Since all correct processes send a message to all in rounds $$1,2, ..., f+1$$, we assume crash failure only, $$p$$'s set $$quiet({f+1})$$ can only contain crashed processes. Since we assume there are $$f$$ failures, there are no more quiet processes to be added to $$quiet(f+1)$$, $$quiet(f+2) = quiet(f+1)$$. At round $$f+2$$, $$p$$ delivers $$SF$$ and then halts by round $$min(f+3,t+1)$$.

**Lemma (Uniform Integrity)**: Every process delivers at most one message, and delivers $$m \neq SF$$ only if $$m$$ was previously broadcast by the sender.

_Proof_: It is clear from the structure of the algorithm that no process delivers more than once. Since only crash failures may occur, we wish to show that if a process delivers $$m \neq SF$$, then the sender sends $$m$$ to some process in the first round.

If some correct process delivers $$m \neq SF$$, then some correct process delivers $$m$$ before round $$t + 1$$.

Suppose some process delivers $$m \neq SF$$ in round $$t+1$$. This implies the existence of $$t+1$$ distinct processes $$\lbrace p_1, ..., p_{t+1} \rbrace$$, such that for $$1 \leq i \leq t+1$$, $$p_i$$ sends $$m$$ in round $$i$$. One of them, $$p_k$$, must be correct. If $$k=1$$, then $$p_k$$ delivered $$m$$ in round 1. If $$2 \leq k \leq t+1$$, then $$p_k$$delivered $$m$$ in round $$k-1$$. In both cases, $$p_k$$ sends $$m$$ in round $$k$$.

If some correct process delivers $$m \neq SF$$, then no correct process ever delivers $$SF$$.

Let $$p$$ be the first correct process to deliver $$m \neq SF$$. $$p$$ delivers $$m$$ in some round $$i$$, $$i < t+1$$. This implies the existence of $$i$$ distinct processes $$\lbrace p_1, ..., p_i \rbrace$$, such that $$1 \leq k \leq i$$, $$p_k$$ sends $$m$$ in round $$k$$.

No process (whether correct or faulty) delivers $$SF$$ before or in round $$i$$.

Suppose the first process to deliver $$SF$$, say process $$q$$, does so by round $$j \leq i$$. This means that the quiet set, $$|quiet(j)| = |quiet(j-1)|$$. In rounds $$1 \leq k \leq j-1$$, $$q$$ did not receive $$m$$ from $$p_k$$. Thus, at the end of round $$j-1$$, $$quiet(j-1)$$includes processes $$\lbrace p_1, ..., p_{j-1} \rbrace$$. Since we assume crash failure only, no processes can be removed from quiet set once they are in there. So $$quite(j) = quite(j-1) = \lbrace p_1, ..., p_{j-1} \rbrace$$.  Since $$p_j$$ also couldn't deliver message in round $$j$$, $$quite(j)$$ should include include $$p_j$$ thus be bigger than $$quite(j-1)$$, which is a contradiction.

Since $$p$$ delivers $$m$$ in round $$i < t+1$$, it sends $$m$$ to all in round $$i+1$$. Furthermore, no process sends $$SF$$ in round $$i+1$$. Thus, in round $$i+1$$, any correct process that has not yet delivered a message receives and delivers $$m$$. Together with the above claim, this implies that no correct process ever delivers $$SF$$.

**Lemma (Agreement)**: If some correct process delivers a message $$m$$ (possibly $$SF$$), then every correct process eventually delivers $$m$$.

_Proof_: By Termination and Uniform Integrity, all correct processes deliver exactly one message. By Uniform Integrity, this message is either $$SF$$ or the sender’s message $$m \neq SF$$. Agreement now follows by the previous lemma.

**Lemma**: This algorithm only works if we assume crash failure only, and it does not work if there exists send omissions.

_Proof_: We have proved using previous lemmas that this algorithm is a valid t-tolerant early-stopping TRB if we only assume crash failures.

Suppose we allow send-omission failures, for arbitrary process $$q$$ has send omission to all other processes. Here is a scenario where this algorithm fails to provide Uniform Integrity. Suppose $$p$$ has send omission failure to all other processes. Sender has send omission except for $$p$$. This means that, $$p$$ will receive message $$m$$ from sender, and deliver $$m$$, whereas other processes will not receive any messages and deliver $$SF$$, which violates uniform integrity.

## A Message-Efficient Early-Stopping TRB Algorithm

### Definition

We assume a system of $$n$$ processes that can communicate through reliable links in a fully connected point-to-point network. Processes have unique ids in the range $$[1, n]$$ which are known a priori to all processes. The computation proceeds in synchronous rounds. Informally, a round is an interval of time where processes first send messages (according to their states), wait to receive messages sent by other processes in the same round, and then change their states accordingly. We consider the following types of process failures:

1. **Crash failures**: A process may fail by halting prematurely. Until it halts, it behaves correctly.
2. **Send-omission failures**: A process may fail not only by crashing, but also by omitting to send some of the messages that it should send.
3. **General-omission failures**: A process may fail by halting or by omitting to send or receive messages.

### Outline of Algorithms

The algorithms in this paper use the rotating coordinator paradigm. A subset of $$t + 1$$ processes cyclically become coordinators for a constant number of rounds each. The sender is the first coordinator and its id is 1. When a process becomes a coordinator, it determines a "consistent" decision value and tries to impose it on the remaining processes. Our algorithms ensure that when a correct process becomes the coordinator, it will succeed in enforcing agreement on the message broadcast. Since at most $$f$$ coordinators can be faulty during an execution of the algorithm, agreement is achieved in $$O(f)$$ rounds. Moreover, in each round, most of the messages axe to or from the coordinator; thus the number of messages sent is $$O(n)$$ per round, and agreement is reached with $$O(fn)$$ messages.

Each process $$p$$ maintains a variable $$estimate_p$$ that represents p's current estimate of the final decision value. Processes can be in one of two states: undecided or decided. A process $$p$$ decides $$v$$ when it sets its variables $$decision_p$$ to $$v$$, and $$state_p$$ to decided. Our algorithms ensure that if a correct process $$p$$ decides $$v$$ (for some $$v$$), then all correct processes eventually decide $$v$$.

```
estimate_p = m if p is the sender, empty otherwise
state_p = undecided

For c in 1,2, ..., t+1:
    Processor c becomes the coordinate for three rounds
    Round 1: All undecided processes p send request to c
        If c does not receive any requests, skip round 2 and round 3
    Round 2: c broadcasts estimate_p
        All undecided processes p that receive estimate_c set estimate_p = estimate_c
    Round 3: c broadcasts decide
        All undecided processes p that receive a decide do
            decision_p = estimate_p
            state_p = decided
```

### Reliable Broadcast for Crash Failures

The above algorithm tolerates crash failures. This algorithm takes $$3f + 3$$ rounds to achieve decision. Each coordinator becomes "active" for three rounds. In the first round undecided processes send a request for "help" to the current coordinator $$c$$ (an undecided coordinator "sends" a request to itself). If the current coordinator $$c$$ does not receive any request, it skips rounds 2 and 3. If $$c$$ receives a request, it broadcasts $$estimate_c$$ in round 2, and decide in round 3.&#x20;

Note that due to the crash failure assumption, if $$c$$ begins to broadcast $$decide$$, then it must have successfully sent $$estimate_c$$ to all. Thus, $$decide$$ is sent only if all processes receive $$estimate_c$$: All future coordinators are guaranteed to have the same message estimate, and will eventually force all processes to decide on it.&#x20;

Let $$T$$ be the round in which the first decide message is received by any process. Let $$p$$ be the coordinator that sent this decide, and let $$estimate_p$$ be the message $$p$$ broadcast in round $$T - 1$$. We can show that all correct processes eventually decide $$estimate_p$$.

**Lemma:** At round $$T-1$$, all processes $$q$$ which did not crash received $$estimate_p$$ and set $$estimate_q$$ to $$estimate_p$$.

**Lemma**: If $$c$$ is a coordinator after $$p$$, and $$c$$ sends $$estimate_c$$, then $$estimate_c = estimate_p$$.

**Lemma**: If coordinator $$c$$ is correct, all processes which have not crashed decide by the end of round $$3c$$.

**Theorem**: Algorithm solves the Reliable Broadcast problem in the presence of crash failures. The correct processes decide by round $$3f + 3$$ after sending at most $$O(fn)$$ messages.

The time complexity of this algorithm can be improved by merging round 1 and round 2. In round 1, any process which  has not decided sends the coordinator a request, and the coordinator broadcasts $$estimate_c$$if it is not decided yet (note that if $$c$$ is decided, all surviving processes must have the same estimate as $$c$$). In round 2, $$c$$ sends $$decide$$if it received a $$request$$in round 1. With this modification, the correct processes decide by round $$2f + 2$$.

Further improvements are possible using pipelining. So far we allow a single coordinator to be alive at a time. The algorithm can be sped up by pipelining its execution so that coordinator $$i+1$$starts only one round after coordinator $$i$$ (while $$i$$ is still alive). Thus coordinator $$c$$ starts in round $$c$$. The resulting algorithm achieves decision in $$f+2$$ rounds.

## Relating Consensus and TRB

### Specification of Consensus

Every process is supposed to propose some value (from a universe $$\mathcal{U}$$ of possible values), and each process must decide on a value such that the following properties hold:&#x20;

• **Agreement**: No two correct processes decide differently.

• **Validity**: If a correct process decides $$v$$, then $$v$$ was proposed by some process.

• **Termination**: Every correct process decides exactly one value.

### Using TRB to Solve Consensus

```
Process i (1 <= i <= n):
    Initially:
        D[1...n] = <null, null, ..., null>
    To propose v_i:
        broadcast v_i [using copy i of TRB]
    To decide:
        while (exists k: D[k] = null) (array is not full)
            if deliver some v_j [from copy j of TRB]
                then D[j] = v_j
        decide first non-SF value in array D
```

### Using Consensus to Solve TRB

The following reduction assumes:

* synchronous rounds
* crash and send-omission failures only
* no link failures

```
Sender s in round 1:
    send m to all; deliver m; halt

Every receiver p not equals s in rounds > 1:
    Run a t-tolerant Consensus algorithm by proposing the message that p received in round 1
    (or SF if p did not receive any message)
    if p decides v, then it delivers v
```

## A $$t+1$$round Lower Bound Proof for Consensus (and therefore also for TRB)

### Background

We consider systems where processes proceed in synchronized rounds: in each round, every process sends messages to other processes, receives all the messages sent to it in that round, and changes state accordingly. When a process crashes in a round, it sends a subset of the messages that it intends to send in that round, and does not execute any subsequent rounds. A correct process is one that never crashes. In the consensus problem, every process starts with some initial value and must make an irrevocable decision on a value such that Agreement, Validity and Termination hold.

### The Proof

The proof shows that any consensus algorithm that tolerates $$t$$ crashes requires $$t+1$$ rounds. The proof proceeds by contradiction. Suppose there is a consensus algorithm $$\mathcal{A}$$ that tolerates up to $$t$$ crashes and always terminates in $$t$$ rounds. In any run of $$\mathcal{A}$$, the configuration at the beginning of round $$t$$ must be univalent. We then obtain a contradiction by constructing a run of $$\mathcal{A}$$ that is bivalent at the beginning of round $$t$$. This run is obtained by starting from a bivalent initial configuration and extending it one round at a time, while maintaining bivalency. Each one-round extension may require the killing of a process

**Theorem**: Consider a synchronous round-based system $$S$$ with $$n$$ processes and at most $$t$$ crash failures such that at most one process crashes in each round. If $$n > t+1$$ then there is no algorithm that solves consensus in $$t$$ rounds in $$S$$.

The proof is by contradiction. Suppose there is an algorithm $$\mathcal{A}$$ that solves consensus in $$t$$ rounds in $$S$$. Without loss of generality, we can assume that $$\mathcal{A}$$ is loquacious, i.e., at every round, each process is supposed to send a message to every process. We consider the configuration of the system $$S$$ at the end of each round (this is also the configuration of the system just before the start of the next round). Such a configuration is just the state of each process (which also indicates the current round number and whether it has crashed in a previous round). Informally, a configuration $$C$$ is $$0-valent$$ $$[1-valent]$$ if starting from $$C$$ the only possible decision value of correct processes is $$0[1]$$; $$C$$ is univalent it is either $$0-valent$$ or $$1-valent$$; $$C$$ is bivalent if it is not univalent. In the following, a $$k$$-round partial run $$r_k$$ denotes an execution of algorithm $$\mathcal{A}$$ up to the end of round $$k$$. Consider the configuration $$C_k$$ at the end of round $$k$$ of partial run $$r_k$$. We say that $$r_k$$ is $$0-valent$$, $$1-valent$$, univalent, or bivalent if $$C_k$$ is $$0-valent$$, $$1-valent$$, univalent, or bivalent.

**Lemma 1**: Any $$(t-1)$$-round partial run $$r_{(t-1)}$$ is univalent.

_Proof_: The proof is by contradiction. Suppose there is a bivalent $$(t-1)$$-round partial run $$r_{t-1}$$. Let $$r^0$$ be the $$t$$-round run obtained by extending $$r_{t-1}$$by one round such that no process crashes in round $$t$$. Without loss of generality assume that all correct processes decide $$0$$ in $$r^0$$. Since partial run $$r_{t-1}$$is bivalent, there is at least one $$t$$-round run $$r^1$$ that all correct processes decide $$1$$.In round $$t$$ of $$r^1$$:

1. Exactly one process $$p$$ must crash (recall that in each run at most one process crashes per round).
2. $$p$$ must fail to send a message to at least one correct process, say $$c$$.

Construct run $$r^{0,1}$$ which is identical to $$r^1$$, except that $$p$$ sends its message to $$c$$. Let $$c'$$ be a process that does not crash in $$r^{0,1}$$ and is different from $$c$$. Such a process must exist since $$n > t + 1$$ implies that there are at least two correct processes in the system. Note that

1. $$c$$ cannot distinguish between $$r^{0,1}$$ and $$r^0$$
2. $$c'$$ cannot distinguish between $$r^{0,1}$$ and $$r^1$$

By 1, $$c$$ decides $$1$$ in $$r^{0,1}$$, which is a violation of agreement property.

**Lemma 2**: There is a bivalent initial configuration.

_Proof_: Suppose, for contradiction, that every initial configuration is univalent. Consider the initial configurations $$C^0$$ and $$C^1$$ such that all processes have initial value 0 and 1, respectively. By the validity property of consensus, $$C^0$$ is $$0-valent$$ and $$C^1$$ is $$1-valent$$. Clearly, there are two initial configurations that differ by the initial value of only one process $$p$$, such that one is $$0-valent$$ and the other is $$1-valent$$. We can easily reach a contradiction by crashing $$p$$ at the beginning of round (before it sends any messages to any process).

**Lemma 3**: There is a bivalent $$(t-1)$$-round partial run $$r_{t-1}$$.

_Proof_: : We show by induction on $$k$$ that for each $$k$$, $$0 \leq k \leq t-1$$, there is a bivalent $$k$$-round partial run $$r_k$$.

Basis: By Lemma 2, there is some bivalent initial configuration $$C_0$$. For $$k=0$$, let $$r_0$$ be the $$0$$-round partial run that ends in $$C_0$$.

Induction Step: Suppose $$0 \leq k < t-1$$. Let $$r_k$$ be a bivalent $$k$$-round partial run. We now show that $$r_k$$ can be extended by one round into a bivalent $$k$$-round partial run $$r_{k+1}$$. Assume, for contradiction, that every one-round extension of $$r_k$$ is univalent.

Let $$r_{k+1}^*$$be the partial run obtained by extending $$r_k$$by one round such that no new crashes occur. Partial run $$r_{k+1}^*$$ is univalent. Without loss of generality assume it is $$1-valent$$. Since $$r_k$$ is bivalent, and every one-round extension of $$r_k$$is univalent, there is at least one-round extension $$r_{k+1}^0$$ of $$r_k$$ that is $$0-valent$$.

Note that $$r_{k+1}^*$$and $$r_{k+1}^0$$must differ in round $$k+1$$(and only in that round). Since round $$k+1$$of $$r_{k+1}^*$$ is failure-free, there must be exactly one process $$p$$ that crashes in round $$k+1$$ of $$r_{k+1}^0$$(recall that in each run, at most one process crashes per round). Since $$p$$ crashes in round $$k+1$$of $$r_{k+1}^0$$ it may fail to send a message to some processes, say to $$q_1, q_2, ..., q_m$$, where $$0 \leq m \leq n$$.

Starting from $$r_{k+1}^0$$, we now define $$(k+1)$$-round partial runs $$r_{k+1}^1, ..., r_{k+1}^m$$ as follows. For every $$j$$, $$1 \leq j \leq m$$, $$r_{k+1}^j$$is identical to $$r_{k+1}^{j-1}$$ except that $$p$$ sends a message to $$q_j$$ before it crashes in round $$k+1$$. Note that for every $$j$$, $$0 \leq j \leq m$$, $$r_{k+1}^j$$is univalent. There are two possible cases:

1. For all $$j$$, $$0 \leq j \leq m$$, $$r_{k+1}^j$$is $$0-valent$$ while $$r_{k=1}^j$$is $$1-valent$$. Extend partial runs $$r_{k+1}^{j-1}$$and $$r_{k+1}^j$$into runs $$r$$ and $$r'$$, by crashing process $$q_j$$at the beginning of round $$k+2$$(before it sends any message in that round), and continuing with no additional crashes. Note that no process except $$q_j$$ can distinguish $$r$$ and $$r'$$ and all correct processes must decide $$0$$ in $$r$$ and $$1$$ in $$r'$$, which is a contradiction.
2. There is a $$j$$, $$1 \leq j \leq m$$, such that $$r_{k+1}^{j-1}$$ is $$0-valent$$ while $$r_{k+1}^j$$ is $$1-valent$$. Extend partial runs $$r_{k+1}^{j-1}$$ and $$r_{k+1}^j$$ into runs $$r$$ and $$r'$$, by crashing process $$q_j$$at the beginning of round $$k+2$$ (before it sends any message in that round), and continuing with no additional crashes. Note that no process except $$q_j$$can distinguish between $$r$$ and $$r'$$, and all correct processes must decide $$0$$ in $$r$$ and $$1$$ in $$r'$$, which is a contradiction.

## Non-Blocking Atomic Commit (NBAC)

### Specification

* **Uniform Agreement**: No two processes decide differently.
* **Validity**:
  * If some processes start with $$0(NO)$$, then no process decides $$1(Commit)$$.
  * If all processes start with $$1(YES)$$, and there are no failures, then no process decides $$0(Abort)$$.
* **Termination**: All nonfaulty processes eventually decide($$Abort$$ or $$Commit$$).

Equivalently, Validity can be written as

If a process decides to ABORT ($$0$$), then

* at least one process vote $$0(NO)$$
* some processes did not vote.

### Solving NBAC Using TRB

```
Each process send its vote to S

if S receives all 1 (YES) with the count of n
    TRB(COMMIT)
else:
    TRB(ABORT)

Each process upon TRB delivery:
    If m = 1, decide COMMIT
    If m = 0 or SF, decide ABORT
```

## Coordinated Attack (CA) Problem and Message Losses

### Specification

In the CA problem, every process starts with an initial input of 0 or 1. Each process can output a decision value $$0$$ or $$1$$ (once a process decides, it cannot "change its mind"). Processes must satisfy the following properties:

* Uniform Agreement: No two processes decide differently.
* Validity:
  * If all processes start with $$0$$, then no process decides $$1$$.
  * If all processes start with $$1$$, and there are no failures, then no process decides $$0$$.
* Termination: All non faulty processes eventually decide.

The CA problem is considered in a simple message-passing system $$\mathcal{S}$$ consisting of only two processes, $$p$$ and $$q$$, connected by a bidirectional communication link such that

* Processes proceed in synchronous rounds
* Processes do not fail
* Messages can be lost

Suppose that the communication link between $$p$$ and $$q$$ is fair, that is, messages can get lost, but if $$p$$ sends a message $$m$$ to $$q$$ infinitely often, then $$q$$ receives $$m$$ infinitely often, and symmetrically for messages sent by $$q$$ to $$p$$. Here is a quiescent algorithm that solves this CA problem, which means that eventually no messages are sent.

### The algorithm

```
start with arbitrary i = 0 or 1
round = 0 if p else 1
required_send = 1
required_receive = 1
while required_send > 0 and required_receive > 0:
    if round % 2 == 0:
        m = TRB_Send(i)
        if m != SF:
            required_send -= 1
    else:
        m = TRB_Receive()
        if m == i:
            required_receive -= 1
        else if m != i and i == 1:
            i = 0
            required_receive -= 1
            if required_send == 0:
                required_send = 1
        else if m != i and i == 0:
            if required_send == 0:
                required_send = 1
    round += 1
decide i; halt
```

The intuition of this algorithm works as follows: In each alternative round, either $$p$$ sends a message to $$q$$ or $$q$$ sends a messages to $$p$$ using TRB. Both $$p$$ and $$q$$ requires to be the receiver at least once and the sender at least once before deciding. If TRB fails to deliver the message, it does not count as successful sent operation. If the received message disagrees with the stored message, previous send does not count as successfully send operation. Instead, if the other processes voted $$0 (NO)$$ and this process intend to vote $$1 (YES)$$, it will change its vote to $$0(NO)$$in order to reach consensus. On the other hand, if this process finds out that the other process intends to vote $$1(YES)$$ which contradicts its vote, it will not change its vote, instead wait for another round to let the other process handle disagreement. Both processes deliver after successfully sent and receive the same vote as the other process.

### The Proof

Since there are only a few scenarios for the initial condition of $$p$$ and $$q$$ for its initial value of $$i$$. We will prove Validity, Uniform and Termination for these scenarios including arbitrary number of failures are involved during message passings.

$$p$$ **and** $$q$$ **both start with** $$0$$ **or** $$1$$**.** \
**Validity and Uniform Agreement**: Since both processes start with the same value, both $$p$$ and $$q$$ never changes their vote. If message is successfully sent at arbitrary odd rounds, $$p$$ decrements $$required\_send$$ to $$0$$ and $$q$$ decrements $$required\_receive$$ to $$0$$. If message is successfully sent at arbitrary even rounds, $$q$$ decrements $$required\_send$$ to $$0$$ and $$p$$ decrements $$required\_receive$$ to $$0$$. The structure of this algorithm requires that at least one message passing is successful on odd rounds and on even rounds. After termination, both processes decide the same value.\
**Termination**: Since we assume that messages between $$p$$ and $$q$$ eventually delivers and both of them do not fail. We wish to prove that both processes deliver at the same round. Since by the program structure, if message is delivered successfully, both processes have the same vote, hence $$required\_receive$$ and $$required\_send$$ decrement at the same round. This means that both processes satisfy the exit condition of the while loop at the same round and halt.

$$p$$ **starts with** $$0$$ **and q starts with** $$1$$**. (The other scenario is symmetric to this scenario.)**\
**Validity**: $$p$$ and $$q$$ do not agree with each other initially, hence validity is satisfied by vacuous truth.\
**Uniform Agreement**: Since $$q$$ starts with $$1$$which disagrees with $$p$$. There are two cases to consider:\
1\. If $$p$$ successfully sends message to $$q$$ first, $$q$$ changes its value to $$0$$, $$required_receive = 0$$ and $$p$$ has $$required\_send = 0$$, then the program does not progress until $$q$$ successfully sends to $$p$$ the agreed value being $$0$$. \
2\. If $$q$$ successfully sends message to $$p$$ first, $$q$$ does not change its value and $$q$$ does not decrement its $$required\_send$$ and $$p$$ does not decrement its $$required\_receive$$. The program proceeds until $$p$$ successfully sends message to $$q$$ and reduces to the first case.\
**Termination**: There are two cases to consider:\
1\.  If $$p$$ successfully sends message to $$q$$ first, $$q$$ changes its value to $$0$$, $$required\_receive = 0$$ and $$p$$ has $$required\_send = 0$$. The program does not progress until $$q$$ successfully sends to $$p$$ the agreed value being $$0$$, such that $$q$$ changes $$required\_send = 0$$ and $$p$$ changes $$required\_receive = 0$$. Both processes deliver at the next round.\
2\. If $$q$$ successfully sends message to $$p$$ first, $$q$$ does not change its value and $$q$$ does not decrement its $$required\_send$$ and $$p$$ does not decrement its $$required\_receive$$. The program proceeds until $$p$$ successfully sends message to $$q$$ and reduces to the first case.

## TRB and Consensus with Arbitrary Failures

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EarnLedger - the on-chain half of EARN (EARN Project Design v2, Section 5)
/// @notice Append-only record of EARN's per-hospital, per-disease trust table. It does two
///         things a plain log does not:
///         1. It locks the history the low-coverage check depends on: each round commits the
///            hash of every hospital's update history, and rounds can only be appended.
///         2. It constrains the coordinator: a commit in which any trust value rises by more
///            than `maxStepBps` over the previous round is rejected, so even whoever runs the
///            aggregation cannot silently boost a hospital. Decreases are unrestricted
///            ("slow up, fast down").
///         Trust is stored in basis points (0..10000), identical to src/ledger/chain.py.
contract EarnLedger {
    uint16 public constant BPS = 10000;
    /// @dev float -> basis-point rounding slack; src/ledger/chain.py uses the same value
    uint16 public constant ROUNDING_BPS = 1;

    address public immutable coordinator;
    uint8 public immutable nClients;
    uint8 public immutable nClasses;
    uint16 public immutable maxStepBps;

    uint32 public round;
    bytes32 public head;
    uint16[] private trust; // nClients * nClasses, row-major (client, class)

    mapping(uint32 => bytes32) public blockHash;
    mapping(uint32 => bytes32) public historyHash;

    event RoundCommitted(
        uint32 indexed round,
        bytes32 blockHash,
        bytes32 historyHash,
        uint16[] trustBps,
        uint8[] coverage
    );

    error NotCoordinator();
    error BadRound(uint32 expected, uint32 got);
    error BadLength();
    error TrustAboveOne(uint256 index, uint16 value);
    error StepTooLarge(uint256 index, uint16 previous, uint16 proposed);

    constructor(uint8 clients, uint8 classes, uint16 stepBps) {
        coordinator = msg.sender;
        nClients = clients;
        nClasses = classes;
        maxStepBps = stepBps;
        trust = new uint16[](uint256(clients) * classes);
    }

    /// @notice Append round `r`. Reverts unless r is the next round, every trust value is in
    ///         [0, 10000], and none rose by more than maxStepBps (+ rounding slack).
    function commitRound(
        uint32 r,
        uint16[] calldata newTrust,
        uint8[] calldata coverage,
        bytes32 histHash
    ) external {
        if (msg.sender != coordinator) revert NotCoordinator();
        if (r != round + 1) revert BadRound(round + 1, r);
        if (newTrust.length != trust.length || coverage.length != nClasses) revert BadLength();

        for (uint256 i = 0; i < newTrust.length; ++i) {
            uint16 t = newTrust[i];
            if (t > BPS) revert TrustAboveOne(i, t);
            uint16 prev = trust[i];
            if (t > prev && t - prev > maxStepBps + ROUNDING_BPS) revert StepTooLarge(i, prev, t);
            if (t != prev) trust[i] = t;
        }

        bytes32 b = keccak256(abi.encode(head, r, newTrust, coverage, histHash));
        head = b;
        round = r;
        blockHash[r] = b;
        historyHash[r] = histHash;
        emit RoundCommitted(r, b, histHash, newTrust, coverage);
    }

    function getTrust() external view returns (uint16[] memory) {
        return trust;
    }

    /// @notice True if `h` is the history hash committed for round `r` - how a verifier checks
    ///         that the history used by the low-coverage check was not rewritten.
    function verifyHistory(uint32 r, bytes32 h) external view returns (bool) {
        return r != 0 && r <= round && historyHash[r] == h;
    }
}

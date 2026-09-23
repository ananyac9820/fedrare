require("@nomicfoundation/hardhat-toolbox");

/** EARN ledger - local chain only. `npx hardhat test`, `npx hardhat run scripts/measure.js`. */
module.exports = {
  solidity: {
    version: "0.8.24",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  networks: {
    hardhat: { chainId: 31337 },
  },
  gasReporter: { enabled: false },
};

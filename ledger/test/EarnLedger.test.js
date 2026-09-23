const { expect } = require("chai");
const { ethers } = require("hardhat");

const K = 6;
const C = 8;
const STEP = 1000; // 0.1 in basis points

const zeros = () => Array(K * C).fill(0);
const cov = () => [6, 6, 3, 3, 5, 4, 3, 2];
const h = (s) => ethers.id(s);

async function deploy() {
  const [coordinator, other] = await ethers.getSigners();
  const ledger = await ethers.deployContract("EarnLedger", [K, C, STEP]);
  return { ledger, coordinator, other };
}

describe("EarnLedger", function () {
  it("appends rounds in order and exposes the trust table", async function () {
    const { ledger } = await deploy();
    const t1 = zeros().map((_, i) => (i % 2 ? 1000 : 0));
    await expect(ledger.commitRound(1, t1, cov(), h("hist1"))).to.emit(ledger, "RoundCommitted");
    expect(await ledger.round()).to.equal(1n);
    expect((await ledger.getTrust()).map(Number)).to.deep.equal(t1);
    expect(await ledger.verifyHistory(1, h("hist1"))).to.equal(true);
    expect(await ledger.verifyHistory(1, h("forged"))).to.equal(false);
  });

  it("allows slow rises of at most one step and falls of any size", async function () {
    const { ledger } = await deploy();
    let t = zeros();
    for (let r = 1; r <= 10; r++) {
      t = t.map((v) => Math.min(10000, v + STEP));
      await ledger.commitRound(r, t, cov(), h(`h${r}`));
    }
    expect((await ledger.getTrust()).map(Number)).to.deep.equal(Array(K * C).fill(10000));
    const halved = t.map((v) => v / 2);
    await ledger.commitRound(11, halved, cov(), h("h11"));
    expect((await ledger.getTrust()).map(Number)[0]).to.equal(5000);
  });

  it("rejects a trust rise faster than the step - even from the coordinator", async function () {
    const { ledger } = await deploy();
    const t = zeros();
    t[2 * C + 6] = 3000; // centre 2, vascular lesion: 0 -> 0.3 in one round
    await expect(ledger.commitRound(1, t, cov(), h("x")))
      .to.be.revertedWithCustomError(ledger, "StepTooLarge")
      .withArgs(2 * C + 6, 0, 3000);
    t[2 * C + 6] = STEP + 1; // rounding slack is allowed
    await ledger.commitRound(1, t, cov(), h("x"));
  });

  it("rejects out-of-order rounds, bad lengths, values above one and other senders", async function () {
    const { ledger, other } = await deploy();
    await expect(ledger.commitRound(2, zeros(), cov(), h("x")))
      .to.be.revertedWithCustomError(ledger, "BadRound").withArgs(1, 2);
    await expect(ledger.commitRound(1, zeros().slice(1), cov(), h("x")))
      .to.be.revertedWithCustomError(ledger, "BadLength");
    const high = zeros();
    high[0] = 10001;
    await expect(ledger.commitRound(1, high, cov(), h("x")))
      .to.be.revertedWithCustomError(ledger, "TrustAboveOne");
    await expect(ledger.connect(other).commitRound(1, zeros(), cov(), h("x")))
      .to.be.revertedWithCustomError(ledger, "NotCoordinator");
  });

  it("chains block hashes so a past round cannot be replaced", async function () {
    const { ledger } = await deploy();
    await ledger.commitRound(1, zeros(), cov(), h("a"));
    const b1 = await ledger.blockHash(1);
    await ledger.commitRound(2, zeros(), cov(), h("b"));
    const coder = ethers.AbiCoder.defaultAbiCoder();
    const expected = ethers.keccak256(
      coder.encode(["bytes32", "uint32", "uint16[]", "uint8[]", "bytes32"], [b1, 2, zeros(), cov(), h("b")])
    );
    expect(await ledger.blockHash(2)).to.equal(expected);
    await expect(ledger.commitRound(1, zeros(), cov(), h("rewrite")))
      .to.be.revertedWithCustomError(ledger, "BadRound");
  });
});

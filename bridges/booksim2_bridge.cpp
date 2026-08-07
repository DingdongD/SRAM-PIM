#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "booksim.hpp"
#include "booksim_config.hpp"
#include "buffer_state.hpp"
#include "credit.hpp"
#include "flit.hpp"
#include "network.hpp"
#include "routefunc.hpp"
#include "trafficmanager.hpp"

TrafficManager* trafficManager = NULL;
bool gPrintActivity = false;
int gK = 0;
int gN = 0;
int gC = 0;
int gNodes = 0;
bool gTrace = false;
ostream* gWatchOut = &std::cerr;

namespace {

int g_bridge_time = 0;

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, '\t')) {
    fields.push_back(field);
  }
  return fields;
}

long long ParseInteger(const std::string& text, const std::string& name) {
  std::size_t consumed = 0;
  long long value = std::stoll(text, &consumed, 10);
  if (consumed != text.size()) {
    throw std::runtime_error("invalid integer for " + name + ": " + text);
  }
  return value;
}

struct Options {
  std::string config_path;
  int ticks_numerator;
  int ticks_denominator;
};

Options ParseOptions(int argc, char** argv) {
  if (argc != 7) {
    throw std::runtime_error(
        "usage: booksim2_bridge --config PATH --ticks-numerator N --ticks-denominator D");
  }
  if (std::string(argv[1]) != "--config" ||
      std::string(argv[3]) != "--ticks-numerator" ||
      std::string(argv[5]) != "--ticks-denominator") {
    throw std::runtime_error("invalid BookSim2 bridge command line");
  }
  int numerator = static_cast<int>(ParseInteger(argv[4], "ticks-numerator"));
  int denominator = static_cast<int>(ParseInteger(argv[6], "ticks-denominator"));
  if (numerator <= 0 || denominator <= 0) {
    throw std::runtime_error("tick ratio must be positive");
  }
  return Options{argv[2], numerator, denominator};
}

struct PacketState {
  int packet_id;
  int src;
  int dst;
  int vc;
  int flits;
  int traffic_class;
  int injected;
  int ejected;
  int hops;
};

struct SourceState {
  bool active;
  int packet_id;
};

class BridgeModule : public Module {
 public:
  BridgeModule() : Module(NULL, "strict_bridge") {}
};

class BookSimBridge {
 public:
  BookSimBridge(const Options& options)
      : m_options(options),
        m_config(),
        m_parent(),
        m_network(NULL),
        m_internal_phase(0),
        m_next_flit_id(0) {
    m_config.ParseFile(options.config_path);
    InitializeRoutingMap(m_config);
    m_network = Network::New(m_config, "strict_network");
    if (m_network == NULL) {
      throw std::runtime_error("BookSim2 failed to create network");
    }
    m_nodes = m_network->NumNodes();
    m_num_vcs = m_config.GetInt("num_vcs");
    m_classes = m_config.GetInt("classes");
    if (m_nodes <= 0 || m_num_vcs <= 0 || m_classes <= 0) {
      throw std::runtime_error("BookSim2 network dimensions are invalid");
    }
    m_sources.resize(m_nodes, SourceState{false, -1});
    for (int source = 0; source < m_nodes; ++source) {
      std::ostringstream name;
      name << "strict_source_" << source;
      std::unique_ptr<BufferState> state(
          new BufferState(m_config, &m_parent, name.str()));
      int vc_alloc_delay = m_config.GetInt("vc_alloc_delay");
      int sw_alloc_delay = m_config.GetInt("sw_alloc_delay");
      int router_latency = m_config.GetInt("routing_delay") +
          (m_config.GetInt("speculative")
               ? std::max(vc_alloc_delay, sw_alloc_delay)
               : vc_alloc_delay + sw_alloc_delay);
      int min_latency = 1 + m_network->GetInject(source)->GetLatency() +
                        router_latency +
                        m_network->GetInjectCred(source)->GetLatency();
      state->SetMinLatency(min_latency);
      m_source_buffers.push_back(std::move(state));
    }
  }

  ~BookSimBridge() {
    delete m_network;
  }

  bool Submit(
      int packet_id,
      int cmodel_cycle,
      int src,
      int dst,
      int vc,
      int flits,
      int traffic_class) {
    ValidateCycle(cmodel_cycle);
    if (src < 0 || src >= m_nodes || dst < 0 || dst >= m_nodes) {
      throw std::runtime_error("BookSim2 endpoint is out of range");
    }
    if (vc < 0 || vc >= m_num_vcs) {
      throw std::runtime_error("BookSim2 VC is out of range");
    }
    if (traffic_class < 0 || traffic_class >= m_classes) {
      throw std::runtime_error("BookSim2 traffic class is out of range");
    }
    if (flits <= 0) {
      throw std::runtime_error("BookSim2 packet must contain at least one flit");
    }
    if (m_packets.find(packet_id) != m_packets.end()) {
      throw std::runtime_error("duplicate BookSim2 packet id");
    }
    if (m_sources[src].active) {
      return false;
    }
    BufferState& buffer = *m_source_buffers[src];
    if (!buffer.IsAvailableFor(vc) || buffer.IsFullFor(vc)) {
      return false;
    }
    buffer.TakeBuffer(vc, packet_id);
    m_packets.emplace(
        packet_id,
        PacketState{packet_id, src, dst, vc, flits, traffic_class, 0, 0, 0});
    m_sources[src] = SourceState{true, packet_id};
    return true;
  }

  std::vector<std::pair<int, int> > Tick(int cmodel_cycle) {
    ValidateCycle(cmodel_cycle);
    m_completed.clear();
    m_internal_phase += m_options.ticks_numerator;
    while (m_internal_phase >= m_options.ticks_denominator) {
      StepNetwork();
      m_internal_phase -= m_options.ticks_denominator;
    }
    ++m_cmodel_cycle;
    return m_completed;
  }

  int Nodes() const { return m_nodes; }
  int NumVCs() const { return m_num_vcs; }
  int Classes() const { return m_classes; }

  bool Empty() const {
    return m_packets.empty();
  }

 private:
  void ValidateCycle(int cycle) const {
    if (cycle != m_cmodel_cycle) {
      throw std::runtime_error("BookSim2 C-model cycle mismatch");
    }
  }

  void StepNetwork() {
    g_bridge_time = m_internal_cycle;

    for (int source = 0; source < m_nodes; ++source) {
      Credit* credit = m_network->ReadCredit(source);
      while (credit != NULL) {
        m_source_buffers[source]->ProcessCredit(credit);
        credit->Free();
        credit = m_network->ReadCredit(source);
      }
    }

    m_network->ReadInputs();
    InjectFlits();
    EjectFlits();
    m_network->Evaluate();
    m_network->WriteOutputs();
    ++m_internal_cycle;
  }

  void InjectFlits() {
    for (int source = 0; source < m_nodes; ++source) {
      if (!m_sources[source].active) {
        continue;
      }
      PacketState& packet = m_packets.at(m_sources[source].packet_id);
      BufferState& buffer = *m_source_buffers[source];
      if (buffer.IsFullFor(packet.vc)) {
        continue;
      }
      Flit* flit = Flit::New();
      flit->id = m_next_flit_id++;
      flit->pid = packet.packet_id;
      flit->vc = packet.vc;
      flit->cl = packet.traffic_class;
      flit->head = packet.injected == 0;
      flit->tail = packet.injected + 1 == packet.flits;
      flit->ctime = m_internal_cycle;
      flit->itime = m_internal_cycle;
      flit->atime = -1;
      flit->record = true;
      flit->src = packet.src;
      flit->dest = packet.dst;
      flit->pri = 0;
      flit->hops = 0;
      flit->watch = false;
      flit->subnetwork = 0;
      flit->intm = -1;
      flit->ph = 0;
      flit->data = NULL;
      buffer.SendingFlit(flit);
      m_network->WriteFlit(flit, source);
      ++packet.injected;
      if (packet.injected == packet.flits) {
        m_sources[source] = SourceState{false, -1};
      }
    }
  }

  void EjectFlits() {
    for (int destination = 0; destination < m_nodes; ++destination) {
      Flit* flit = m_network->ReadFlit(destination);
      while (flit != NULL) {
        std::map<int, PacketState>::iterator it = m_packets.find(flit->pid);
        if (it == m_packets.end()) {
          throw std::runtime_error("BookSim2 ejected unknown packet");
        }
        PacketState& packet = it->second;
        if (flit->dest != destination || flit->vc != packet.vc) {
          throw std::runtime_error("BookSim2 ejected flit metadata mismatch");
        }
        Credit* credit = Credit::New();
        credit->vc.insert(flit->vc);
        m_network->WriteCredit(credit, destination);
        ++packet.ejected;
        packet.hops = std::max(packet.hops, flit->hops);
        bool tail = flit->tail;
        flit->Free();
        if (tail) {
          if (packet.ejected != packet.flits) {
            throw std::runtime_error("BookSim2 tail ejected before the full packet");
          }
          m_completed.push_back(std::make_pair(packet.packet_id, packet.hops));
          m_packets.erase(it);
        }
        flit = m_network->ReadFlit(destination);
      }
    }
  }

  Options m_options;
  BookSimConfig m_config;
  BridgeModule m_parent;
  Network* m_network;
  int m_nodes = 0;
  int m_num_vcs = 0;
  int m_classes = 0;
  int m_cmodel_cycle = 0;
  int m_internal_cycle = 0;
  int m_internal_phase;
  int m_next_flit_id;
  std::vector<std::unique_ptr<BufferState> > m_source_buffers;
  std::vector<SourceState> m_sources;
  std::map<int, PacketState> m_packets;
  std::vector<std::pair<int, int> > m_completed;
};

}  // namespace

int GetSimTime() {
  return g_bridge_time;
}

class Stats;
Stats* GetStats(const std::string&) {
  return NULL;
}

int main(int argc, char** argv) {
  const Options options = ParseOptions(argc, argv);
  BookSimBridge bridge(options);
  std::cout << "HELLO\tbooksim2\t1" << std::endl;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::vector<std::string> fields = SplitTabs(line);
    if (fields.empty()) {
      throw std::runtime_error("empty BookSim2 bridge command");
    }
    if (fields[0] == "INFO") {
      if (fields.size() != 1) {
        throw std::runtime_error("BookSim2 INFO takes no arguments");
      }
      std::cout << "INFO_RESULT\t" << bridge.Nodes() << "\t" << bridge.NumVCs()
                << "\t" << bridge.Classes() << std::endl;
      continue;
    }

    if (fields[0] == "SUBMIT") {
      if (fields.size() != 8) {
        throw std::runtime_error("BookSim2 SUBMIT requires 7 arguments");
      }
      int packet_id = static_cast<int>(ParseInteger(fields[1], "packet_id"));
      int cycle = static_cast<int>(ParseInteger(fields[2], "cycle"));
      int src = static_cast<int>(ParseInteger(fields[3], "src"));
      int dst = static_cast<int>(ParseInteger(fields[4], "dst"));
      int vc = static_cast<int>(ParseInteger(fields[5], "vc"));
      int flits = static_cast<int>(ParseInteger(fields[6], "flits"));
      int traffic_class = static_cast<int>(ParseInteger(fields[7], "traffic_class"));
      bool accepted = bridge.Submit(packet_id, cycle, src, dst, vc, flits, traffic_class);
      std::cout << "SUBMIT_RESULT\t" << packet_id << "\t" << (accepted ? 1 : 0) << std::endl;
      continue;
    }
    if (fields[0] == "TICK") {
      if (fields.size() != 2) {
        throw std::runtime_error("BookSim2 TICK requires one argument");
      }
      int cycle = static_cast<int>(ParseInteger(fields[1], "cycle"));
      std::vector<std::pair<int, int> > completed = bridge.Tick(cycle);
      std::cout << "TICK_RESULT\t" << cycle << "\t";
      if (completed.empty()) {
        std::cout << "-";
      } else {
        for (std::size_t index = 0; index < completed.size(); ++index) {
          if (index != 0) {
            std::cout << ",";
          }
          std::cout << completed[index].first << ":" << completed[index].second;
        }
      }
      std::cout << std::endl;
      continue;
    }
    if (fields[0] == "CLOSE") {
      if (fields.size() != 1) {
        throw std::runtime_error("BookSim2 CLOSE takes no arguments");
      }
      if (!bridge.Empty()) {
        throw std::runtime_error("BookSim2 CLOSE while packets are still in flight");
      }
      std::cout << "CLOSED" << std::endl;
      return 0;
    }
    throw std::runtime_error("unknown BookSim2 bridge command: " + fields[0]);
  }
  throw std::runtime_error("BookSim2 bridge stdin closed before CLOSE");
}

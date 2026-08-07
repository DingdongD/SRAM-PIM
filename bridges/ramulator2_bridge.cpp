#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

struct Args {
  std::string config;
  long long ticks_numerator;
  long long ticks_denominator;
};

Args parse_args(int argc, char** argv) {
  if (argc != 7) {
    throw std::runtime_error(
        "usage: ramulator2_bridge --config PATH --ticks-numerator N --ticks-denominator D");
  }
  if (std::string(argv[1]) != "--config" ||
      std::string(argv[3]) != "--ticks-numerator" ||
      std::string(argv[5]) != "--ticks-denominator") {
    throw std::runtime_error("invalid ramulator2_bridge arguments");
  }
  Args args{argv[2], std::stoll(argv[4]), std::stoll(argv[6])};
  if (args.ticks_numerator <= 0 || args.ticks_denominator <= 0) {
    throw std::runtime_error("tick ratio must be positive");
  }
  return args;
}

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::stringstream stream(line);
  while (std::getline(stream, field, '\t')) {
    fields.push_back(field);
  }
  return fields;
}

std::string join_ids(const std::vector<long long>& values) {
  if (values.empty()) {
    return "-";
  }
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << values[index];
  }
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  auto config = Ramulator::Config::parse_config_file(args.config);
  auto* frontend = Ramulator::Factory::create_frontend(config);
  auto* memory_system = Ramulator::Factory::create_memory_system(config);
  frontend->connect_memory_system(memory_system);
  memory_system->connect_frontend(frontend);

  std::vector<long long> completed;
  long long phase = 0;
  long long cmodel_cycle = 0;
  std::cout << "HELLO\tramulator2\t1" << std::endl;

  std::string line;
  while (std::getline(std::cin, line)) {
    const auto fields = split_tabs(line);
    if (fields.empty()) {
      throw std::runtime_error("empty command");
    }

    if (fields[0] == "INFO") {
      if (fields.size() != 1) {
        throw std::runtime_error("INFO expects no arguments");
      }
      std::cout << "INFO_RESULT\t" << memory_system->get_tx_bytes()
                << "\t" << Ramulator::Request::Type::Read
                << "\t" << Ramulator::Request::Type::Write << std::endl;
      continue;
    }

    if (fields[0] == "SUBMIT") {
      if (fields.size() != 7) {
        throw std::runtime_error("SUBMIT expects six arguments");
      }
      const long long request_id = std::stoll(fields[1]);
      const long long cycle = std::stoll(fields[2]);
      const int request_type = std::stoi(fields[3]);
      const auto address = static_cast<Ramulator::Addr_t>(std::stoull(fields[4]));
      const int source_id = std::stoi(fields[5]);
      const int nbytes = std::stoi(fields[6]);
      if (cycle != cmodel_cycle) {
        throw std::runtime_error("SUBMIT cycle does not match bridge cycle");
      }
      if (nbytes <= 0 || nbytes > memory_system->get_tx_bytes()) {
        throw std::runtime_error("external request size exceeds Ramulator transaction size");
      }
      const bool accepted = frontend->receive_external_requests(
          request_type,
          address,
          source_id,
          [request_id, &completed](Ramulator::Request&) { completed.push_back(request_id); },
          nbytes);
      std::cout << "SUBMIT_RESULT\t" << request_id << '\t' << (accepted ? 1 : 0) << std::endl;
      continue;
    }

    if (fields[0] == "TICK") {
      if (fields.size() != 2) {
        throw std::runtime_error("TICK expects one argument");
      }
      const long long cycle = std::stoll(fields[1]);
      if (cycle != cmodel_cycle) {
        throw std::runtime_error("TICK cycle does not match bridge cycle");
      }
      phase += args.ticks_numerator;
      while (phase >= args.ticks_denominator) {
        memory_system->tick();
        phase -= args.ticks_denominator;
      }
      std::cout << "TICK_RESULT\t" << cycle << '\t' << join_ids(completed) << std::endl;
      completed.clear();
      ++cmodel_cycle;
      continue;
    }

    if (fields[0] == "CLOSE") {
      if (fields.size() != 1) {
        throw std::runtime_error("CLOSE expects no arguments");
      }
      frontend->finalize();
      memory_system->finalize();
      delete frontend;
      delete memory_system;
      std::cout << "CLOSED" << std::endl;
      return 0;
    }

    throw std::runtime_error("unknown ramulator2_bridge command");
  }

  throw std::runtime_error("stdin closed before CLOSE");
}
